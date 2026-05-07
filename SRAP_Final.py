import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as M
import torchvision.transforms as T


class PEE_RAP_ATTACK(object):

    def __init__(self, attack_config):

        self.image_size = attack_config["image_size"]
        self.block_size = attack_config["block_size"]
        self.mask_num = attack_config["mask_num"]
        self.quant_step = attack_config["quant_step"]
        self.attack_scale = 64 // self.quant_step
        self.bit_len = math.ceil(math.log2(2 * self.attack_scale))
        self.max_epoch = attack_config["max_epoch"]
        self.device = torch.device(attack_config["device"])
        self.info_len = (
            1 * 3 * self.bit_len * self.block_size * self.block_size + 15 * 15
        )
        cls_model = getattr(M, attack_config["cls_model"])(weights="DEFAULT")
        cls_model.eval()
        self.target_model = nn.Sequential(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            cls_model,
        )
        self.target_model.to(self.device)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)

        print("target model {} load done".format(attack_config["cls_model"]))

    def attack(self, image: torch.Tensor, label: int):
        if image.size(0) == 4:
            image = image[:3]
        elif image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        image = image.unsqueeze(0)
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )

        image = image.float().to(self.device)

        if image.max() <= 1.0:
            image = image * 255.0
        image = image.clamp(0, 255)

        label = torch.tensor([label], device=self.device, dtype=torch.long)

        # Compute the input gradient.
        image_for_grad = image.detach().clone().requires_grad_(True)
        output = self.target_model(image_for_grad / 255.0)
        cls_loss = F.cross_entropy(output, label)
        grad = torch.autograd.grad(cls_loss, image_for_grad)[0].detach()

        # Generate the attack mask.
        mask = self.generate_most_grad_mask(grad).detach()

        # Scale the image regions covered by the mask.
        offset_image = scale_image(image, mask)

        # Compute the pixel range only within the masked region.
        masked_pixels = offset_image[mask.bool()]
        upper_bound = (255 - masked_pixels).min()
        lower_bound = masked_pixels.min()

        b, c, _, _ = image.shape

        perturbation_patch = torch.randint(
            low=-self.attack_scale // 2,
            high=self.attack_scale // 2,
            size=(b, c, self.block_size, self.block_size),
            device=self.device,
            dtype=torch.float32
        )

        adversarial_sample = None
        adv_label = None

        for epoch in range(self.max_epoch):
            perturbation_patch = perturbation_patch.detach().requires_grad_(True)

            perturbation = perturbation_patch.repeat(
                1,
                1,
                self.image_size // self.block_size,
                self.image_size // self.block_size
            ) * self.quant_step

            adversarial_sample = (offset_image + perturbation * mask).clamp(0, 255)

            adv_output = self.target_model(adversarial_sample / 255.0)
            adv_label = adv_output.argmax(dim=1)

            adv_loss = F.cross_entropy(adv_output, label)
            adv_grad = torch.autograd.grad(adv_loss, perturbation_patch)[0]

            # Early stopping for untargeted attacks.
            if adv_label.item() != label.item():
                break

            with torch.no_grad():
                perturbation_patch = perturbation_patch + adv_grad.sign()
                perturbation_patch = perturbation_patch.clamp(
                    -self.attack_scale + 1,
                    self.attack_scale - 1
                )
                perturbation_patch = self.quant(
                    perturbation_patch,
                    upper_bound=upper_bound,
                    lower_bound=lower_bound
                )

        hide_info = []
        hide_info.extend(self.encode_mask(mask))
        hide_info.extend(self.patch2bin(perturbation_patch.detach()))

        hide_adv_image = self.hide_patch(
            adv_image=adversarial_sample.detach(),
            patch_info=hide_info
        )

        return hide_adv_image.detach(), adv_label.detach(), mask.detach()

    def recover(self, hide_adv_image):

        recover_adv_images, extract_patches, extract_mask = self.extract_patch(
            adv_image=hide_adv_image, patch_info_len=self.info_len
        )
        extract_patches = extract_patches.to(torch.float32)
        extract_mask = extract_mask.to(self.device)
        extract_patches = extract_patches.to(self.device)

        with torch.no_grad():
            perturbation = (
                extract_mask
                * (
                    extract_patches.repeat(
                        1,
                        1,
                        self.image_size // self.block_size,
                        self.image_size // self.block_size,
                    )
                )
                * self.quant_step
            )
            restore_clean_image = recover_adv_images - perturbation
            recover_clean_image = recover_scale_image(restore_clean_image, extract_mask)

            return recover_clean_image.detach().clone()

    def quant(self, perturbation, upper_bound, lower_bound):
        min_q = torch.floor(-lower_bound / self.quant_step)
        max_q = torch.floor(upper_bound / self.quant_step)

        perturbation = perturbation.clamp(min=min_q.item(), max=max_q.item())
        return perturbation.detach()

    def dec2bin_list(self, dec_num: int):
        """
        Convert a decimal number to a fixed-length binary representation and return it as a list.

        dec_num: Decimal number to be converted.
        bit_len: Maximum fixed binary length.
        """

        assert dec_num < 2**self.bit_len, print(dec_num)
        bit_str = bin(dec_num)[2:]
        padding_len = self.bit_len - len(bit_str)
        for _ in range(padding_len):
            bit_str = "0" + bit_str
        bit_list = [int(i) for i in bit_str]
        return bit_list

    def bin_list2dec(self, bin_list: list):
        """
        Convert a binary list to a decimal integer.

        bin_list: Binary list.
        bit_len: Maximum binary length.
        """
        assert len(bin_list) == self.bit_len
        bin_str = ""
        for b in bin_list:
            bin_str += str(b)
        y = int(bin_str, base=2)

        return y

    def patch2bin(self, patch: torch.Tensor):
        values = (patch.reshape(-1).long() + self.attack_scale)

        shifts = torch.arange(
            self.bit_len - 1,
            -1,
            -1,
            device=values.device
        )

        bits = ((values[:, None] >> shifts) & 1).reshape(-1)
        return bits.detach().cpu().tolist()

    def bin2patch(self, bin_list: list):
        bits = torch.tensor(bin_list, dtype=torch.long)
        bits = bits.view(-1, self.bit_len)

        shifts = torch.arange(self.bit_len - 1, -1, -1, dtype=torch.long)
        values = (bits * (1 << shifts)).sum(dim=1)

        patch = values.view(1, 3, self.block_size, self.block_size).float()
        return patch

    def generate_most_grad_mask(self, grad: torch.Tensor):
        """
        grad: [B, C, H, W]
        return: [B, C, H, W], binary mask
        """
        b, c, h, w = grad.shape
        bs = self.block_size

        # Sum of absolute gradients in each block: [B, C, H/bs, W/bs].
        block_score = F.avg_pool2d(
            grad.abs(),
            kernel_size=bs,
            stride=bs
        ) * (bs * bs)

        # Merge channels to obtain the total gradient magnitude for each spatial block: [B, H/bs, W/bs].
        block_score = block_score.sum(dim=1)

        num_blocks_h, num_blocks_w = block_score.shape[-2:]
        flat_score = block_score.view(b, -1)

        topk_idx = torch.topk(flat_score, k=self.mask_num, dim=1).indices

        flat_mask = torch.zeros_like(flat_score)
        flat_mask.scatter_(1, topk_idx, 1.0)

        block_mask = flat_mask.view(b, 1, num_blocks_h, num_blocks_w)

        # Expand the block-level mask back to the original image size.
        mask = block_mask.repeat_interleave(bs, dim=2).repeat_interleave(bs, dim=3)

        # Prevent shape mismatch when image_size is not divisible by block_size.
        mask = mask[:, :, :h, :w]

        # Expand the mask to 3 channels.
        mask = mask.expand(b, c, h, w)

        return mask

    def hide_patch(self, adv_image: torch.Tensor, patch_info: list):
        split = len(patch_info) // 3

        r_info = patch_info[:split]
        g_info = patch_info[split:2 * split]
        b_info = patch_info[2 * split:]

        r_hide = pee_embed(adv_image[:, 0:1], r_info)
        g_hide = pee_embed(adv_image[:, 1:2], g_info)
        b_hide = pee_embed(adv_image[:, 2:3], b_info)

        return torch.cat([r_hide, g_hide, b_hide], dim=1).detach()

    def extract_patch(self, adv_image: torch.Tensor, patch_info_len: int):

        r_image = adv_image[:, 0:1, ...]
        g_image = adv_image[:, 1:2, ...]
        b_image = adv_image[:, 2:3, ...]
        info = []

        r_recover_image, r_info = pee_extract(r_image, int(patch_info_len / 3))
        g_recover_image, g_info = pee_extract(g_image, int(patch_info_len / 3))
        b_recover_image, b_info = pee_extract(b_image, int(patch_info_len / 3))

        recover_image = torch.zeros_like(adv_image)
        recover_image[:, 0:1, ...] = r_recover_image
        recover_image[:, 1:2, ...] = g_recover_image
        recover_image[:, 2:3, ...] = b_recover_image

        info.extend(r_info)
        info.extend(g_info)
        info.extend(b_info)
        print(len(info))
        mask_info = info[:225]
        mask = self.decode_mask(mask_info)
        patch_info = info[225:]
        patches = self.bin2patch(patch_info)
        patches = patches - self.attack_scale
        return recover_image.detach().clone(), patches, mask

    def encode_mask(self, mask):
        mask = mask[:, 0:1, ...]
        mask = F.interpolate(mask, size=(15, 15))
        mask = mask.reshape(-1)
        mask = mask.tolist()
        mask = [int(x) for x in mask]
        return mask

    def decode_mask(self, compressed_mask):

        mask = torch.tensor(compressed_mask, dtype=torch.float32)
        mask = mask.reshape(1, 1, 15, 15)
        mask = F.interpolate(mask, size=(self.image_size, self.image_size))
        mask = torch.cat([mask, mask, mask], dim=1)
        return mask


def compute_predict_error(image: torch.Tensor):
    device = image.device
    dtype = image.dtype

    kernel = torch.zeros(1, 1, 3, 3, device=device, dtype=dtype)
    kernel[..., 0, 1] = 0.25
    kernel[..., 1, 0] = 0.25
    kernel[..., 2, 1] = 0.25
    kernel[..., 1, 2] = 0.25

    center_kernel = torch.zeros_like(kernel)
    center_kernel[..., 1, 1] = 1

    predict_image_1 = torch.ceil(
        F.conv2d(image, kernel, stride=2, padding=0)
    )
    predict_image_2 = torch.ceil(
        F.conv2d(image[..., 1:-1, 1:-1], kernel, stride=2, padding=0)
    )

    grid_image_1 = F.conv2d(image, center_kernel, stride=2, padding=0)
    grid_image_2 = F.conv2d(image[..., 1:-1, 1:-1], center_kernel, stride=2, padding=0)

    predict_error_1 = grid_image_1 - predict_image_1
    predict_error_1 = F.conv_transpose2d(predict_error_1, center_kernel, stride=2)

    predict_error_2 = grid_image_2 - predict_image_2
    predict_error_2 = F.conv_transpose2d(predict_error_2, center_kernel, stride=2)
    predict_error_2 = F.pad(predict_error_2, (1, 1, 1, 1), mode="constant", value=0)

    return predict_error_1 + predict_error_2


def pee_embed(image: torch.Tensor, info=None):
    if info is None:
        info = []

    with torch.no_grad():
        predict_error = compute_predict_error(image)

        flat_error = predict_error.flatten()
        unique_values, counts = torch.unique(flat_error, return_counts=True)
        max_bin = unique_values[counts.argmax()]
        a = max_bin - 1
        b = max_bin + 1

        embed_image = image.clone()

        # Process histogram shifting first.
        embed_image = torch.where(predict_error > b, embed_image + 1, embed_image)
        embed_image = torch.where(predict_error < a, embed_image - 1, embed_image)

        # Find embeddable positions where e == a.
        candidate_mask = (predict_error == a)
        candidate_idx = candidate_mask.flatten().nonzero(as_tuple=False).flatten()

        n = min(len(info), candidate_idx.numel())
        if n > 0:
            info_tensor = torch.tensor(
                info[:n],
                device=image.device,
                dtype=image.dtype
            )

            flat_embed = embed_image.flatten()
            flat_embed[candidate_idx[:n]] -= info_tensor
            embed_image = flat_embed.view_as(image)

        return embed_image.clamp(0, 255)


def pee_extract(image: torch.Tensor, info_len=0):
    with torch.no_grad():
        predict_error = compute_predict_error(image)

        flat_error = predict_error.flatten()
        unique_values, counts = torch.unique(flat_error, return_counts=True)
        max_bin = unique_values[counts.argmax()]
        a = max_bin - 1
        b = max_bin + 1

        result_image = image.clone()

        result_image = torch.where(predict_error > b, result_image - 1, result_image)
        result_image = torch.where(predict_error < a, result_image + 1, result_image)

        one_mask = (predict_error == (a - 1)) | (predict_error == (b + 1))
        zero_mask = (predict_error == a) | (predict_error == b)
        bit_mask = one_mask | zero_mask

        bit_idx = bit_mask.flatten().nonzero(as_tuple=False).flatten()
        bit_idx = bit_idx[:info_len]

        one_flat = one_mask.flatten()
        bits = one_flat[bit_idx].to(torch.int64).tolist()

        return result_image.clamp(0, 255), bits


def scale_image(image, mask):
    scale_image = image.detach().clone()
    scale_image = ((scale_image - 127) // 2) + 127
    result_image = image * (1 - mask) + scale_image * mask
    return result_image.detach().clone()


def recover_scale_image(scale_image, mask):

    image = scale_image.detach().clone()
    image = ((image - 127) * 2 + 127).clamp(0, 255)
    result_image = image * mask + scale_image * (1 - mask)
    return result_image.detach().clone()
