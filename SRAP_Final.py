import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # mask is encoded as 15 x 15 = 225 bits.
        # patch info length = 3 channels * block_size * block_size * bit_len.
        self.mask_info_len = 15 * 15
        self.patch_info_len = 3 * self.bit_len * self.block_size * self.block_size
        self.info_len = self.mask_info_len + self.patch_info_len

        cls_model = getattr(M, attack_config["cls_model"])(weights="DEFAULT")
        cls_model.eval()

        self.target_model = nn.Sequential(
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            cls_model,
        )

        self.target_model.to(self.device)
        self.target_model.eval()

        for p in self.target_model.parameters():
            p.requires_grad_(False)

        print("target model {} load done".format(attack_config["cls_model"]))

    def attack(self, image: torch.Tensor, label: int):
        """
        image:
            [C, H, W], value range [0, 255] or [0, 1]

        label:
            ImageNet class index.
        """
        if image.dim() != 3:
            raise ValueError(f"Expected image shape [C, H, W], got {image.shape}")

        if image.size(0) == 4:
            image = image[:3]
        elif image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        elif image.size(0) != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {image.size(0)}")

        image = image.unsqueeze(0)

        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        image = image.round().clamp(0, 255)

        image = image.float().to(self.device)

        if image.max() <= 1.0:
            image = image * 255.0

        image = image.clamp(0, 255)

        label = torch.tensor(
            [label],
            device=self.device,
            dtype=torch.long,
        )

        # Compute input gradient.
        image_for_grad = image.detach().clone().requires_grad_(True)
        output = self.target_model(image_for_grad / 255.0)
        cls_loss = F.cross_entropy(output, label)
        grad = torch.autograd.grad(cls_loss, image_for_grad)[0].detach()

        # Generate attack mask.
        mask = self.generate_most_grad_mask(grad).detach()

        # Important:
        # Canonicalize mask before attack, so the attack-stage mask is exactly
        # the same kind of mask that can be reconstructed during recovery.
        mask = self.decode_mask(self.encode_mask(mask)).to(self.device)

        # Scale the image regions covered by the canonical mask.
        offset_image = scale_image(image, mask)

        # Compute safe perturbation range inside masked region.
        masked_pixels = offset_image[mask.bool()]
        upper_bound = (255 - masked_pixels).min()
        lower_bound = masked_pixels.min()

        b, c, _, _ = image.shape

        perturbation_patch = torch.randint(
            low=-self.attack_scale // 2,
            high=self.attack_scale // 2,
            size=(b, c, self.block_size, self.block_size),
            device=self.device,
            dtype=torch.float32,
        )

        adversarial_sample = None
        adv_label = None

        for epoch in range(self.max_epoch):
            perturbation_patch = perturbation_patch.detach().requires_grad_(True)

            perturbation, placement_mask = place_patch_by_mask(
                mask=mask,
                patch=perturbation_patch,
                block_size=self.block_size,
                quant_step=self.quant_step,
            )

            adversarial_sample = (offset_image + perturbation).clamp(0, 255)
            adv_output = self.target_model(adversarial_sample / 255.0)
            adv_label = adv_output.argmax(dim=1)

            adv_loss = F.cross_entropy(adv_output, label)
            adv_grad = torch.autograd.grad(adv_loss, perturbation_patch)[0]

            # Early stopping for untargeted attack.
            if adv_label.item() != label.item():
                break

            with torch.no_grad():
                perturbation_patch = perturbation_patch + adv_grad.sign()

                perturbation_patch = perturbation_patch.clamp(
                    -self.attack_scale + 1,
                    self.attack_scale - 1,
                )

                perturbation_patch = self.quant(
                    perturbation_patch,
                    upper_bound=upper_bound,
                    lower_bound=lower_bound,
                )

        hide_info = []
        hide_info.extend(self.encode_mask(mask))
        hide_info.extend(self.patch2bin(perturbation_patch.detach()))

        if len(hide_info) != self.info_len:
            raise RuntimeError(
                f"Internal info length mismatch: "
                f"expected {self.info_len}, got {len(hide_info)}"
            )

        # PEE 信息隐藏必须在整数像素域中进行
        adversarial_sample = adversarial_sample.detach().round().clamp(0, 255)

        # 重新计算量化后的对抗样本预测类别
        with torch.no_grad():
            adv_output = self.target_model(adversarial_sample / 255.0)
            adv_label = adv_output.argmax(dim=1)

        debug_mask = self.decode_mask(self.encode_mask(mask)).to(mask.device)
        print("[DEBUG] mask diff:",
            (debug_mask - mask).abs().sum().item())

        debug_perturbation, debug_place = place_patch_by_mask(
            mask=mask,
            patch=perturbation_patch.detach(),
            block_size=self.block_size,
            quant_step=self.quant_step,
        )

        print("[DEBUG] placement pixels:",
            debug_place[:, 0:1].sum().item())
        hide_adv_image = self.hide_patch(
            adv_image=adversarial_sample,
            patch_info=hide_info,
        )

        return hide_adv_image.detach(), adv_label.detach(), mask.detach()

    def recover(self, hide_adv_image):
        recover_adv_images, extract_patches, extract_mask = self.extract_patch(
            adv_image=hide_adv_image,
            patch_info_len=self.info_len,
        )

        extract_patches = extract_patches.to(torch.float32).to(self.device)
        extract_mask = extract_mask.to(self.device)
        print("[DEBUG] extract mask pixels:",
            extract_mask[:, 0:1].sum().item())

        print("[DEBUG] extract patch min/max:",
            extract_patches.min().item(),
            extract_patches.max().item())
        with torch.no_grad():
            perturbation, placement_mask = place_patch_by_mask(
                mask=extract_mask,
                patch=extract_patches,
                block_size=self.block_size,
                quant_step=self.quant_step,
            )

            restore_clean_image = recover_adv_images - perturbation
            recover_clean_image = recover_scale_image(
                restore_clean_image,
                extract_mask,
            )

            return recover_clean_image.detach().clone()

    def quant(self, perturbation, upper_bound, lower_bound):
        min_q = torch.floor(-lower_bound / self.quant_step)
        max_q = torch.floor(upper_bound / self.quant_step)

        perturbation = perturbation.clamp(
            min=min_q.item(),
            max=max_q.item(),
        )

        return perturbation.detach()

    def dec2bin_list(self, dec_num: int):
        assert dec_num < 2**self.bit_len, print(dec_num)

        bit_str = bin(dec_num)[2:]
        padding_len = self.bit_len - len(bit_str)

        for _ in range(padding_len):
            bit_str = "0" + bit_str

        bit_list = [int(i) for i in bit_str]

        return bit_list

    def bin_list2dec(self, bin_list: list):
        assert len(bin_list) == self.bit_len

        bin_str = ""
        for b in bin_list:
            bin_str += str(b)

        y = int(bin_str, base=2)

        return y

    def patch2bin(self, patch: torch.Tensor):
        """
        Convert perturbation patch to binary list.

        patch:
            [1, 3, block_size, block_size]
        """
        values = patch.reshape(-1).long() + self.attack_scale

        shifts = torch.arange(
            self.bit_len - 1,
            -1,
            -1,
            device=values.device,
        )

        bits = ((values[:, None] >> shifts) & 1).reshape(-1)

        return bits.detach().cpu().tolist()

    def bin2patch(self, bin_list: list):
        """
        Convert binary list back to perturbation patch.

        Return:
            [1, 3, block_size, block_size]
        """
        expected_len = 3 * self.block_size * self.block_size * self.bit_len

        if len(bin_list) < expected_len:
            raise RuntimeError(
                f"Patch bitstream is too short: "
                f"expected {expected_len}, got {len(bin_list)}"
            )

        if len(bin_list) > expected_len:
            bin_list = bin_list[:expected_len]

        bits = torch.tensor(bin_list, dtype=torch.long)
        bits = bits.view(-1, self.bit_len)

        shifts = torch.arange(
            self.bit_len - 1,
            -1,
            -1,
            dtype=torch.long,
        )

        values = (bits * (1 << shifts)).sum(dim=1)

        patch = values.view(
            1,
            3,
            self.block_size,
            self.block_size,
        ).float()

        return patch

    def generate_most_grad_mask(self, grad: torch.Tensor):
        """
        Select top-k high-gradient blocks.

        grad:
            [B, C, H, W]

        return:
            [B, C, H, W], binary mask.
        """
        b, c, h, w = grad.shape
        bs = self.block_size

        block_score = F.avg_pool2d(
            grad.abs(),
            kernel_size=bs,
            stride=bs,
        ) * (bs * bs)

        block_score = block_score.sum(dim=1)

        num_blocks_h, num_blocks_w = block_score.shape[-2:]
        flat_score = block_score.view(b, -1)

        total_blocks = flat_score.size(1)
        k = min(self.mask_num, total_blocks)

        topk_idx = torch.topk(flat_score, k=k, dim=1).indices

        flat_mask = torch.zeros_like(flat_score)
        flat_mask.scatter_(1, topk_idx, 1.0)

        block_mask = flat_mask.view(
            b,
            1,
            num_blocks_h,
            num_blocks_w,
        )

        mask = block_mask.repeat_interleave(bs, dim=2).repeat_interleave(bs, dim=3)

        # Pad when block_size does not divide image_size.
        mh, mw = mask.shape[-2:]

        if mh < h or mw < w:
            mask = F.pad(
                mask,
                (
                    0, max(w - mw, 0),
                    0, max(h - mh, 0),
                ),
                mode="constant",
                value=0,
            )

        mask = mask[:, :, :h, :w]
        mask = mask.expand(b, c, h, w)

        return mask

    def hide_patch(self, adv_image: torch.Tensor, patch_info: list):
        """
        Embed recovery information into RGB channels.

        adv_image:
            [1, 3, H, W]

        patch_info:
            list of bits.
        """
        if len(patch_info) != self.info_len:
            raise RuntimeError(
                f"patch_info length mismatch: "
                f"expected {self.info_len}, got {len(patch_info)}"
            )

        split_1 = len(patch_info) // 3
        split_2 = 2 * len(patch_info) // 3

        r_info = patch_info[:split_1]
        g_info = patch_info[split_1:split_2]
        b_info = patch_info[split_2:]

        r_hide = pee_embed(adv_image[:, 0:1], r_info)
        g_hide = pee_embed(adv_image[:, 1:2], g_info)
        b_hide = pee_embed(adv_image[:, 2:3], b_info)

        return torch.cat([r_hide, g_hide, b_hide], dim=1).detach()

    def extract_patch(self, adv_image: torch.Tensor, patch_info_len: int):
        """
        Extract recovery information from RGB channels.
        """
        r_image = adv_image[:, 0:1, ...]
        g_image = adv_image[:, 1:2, ...]
        b_image = adv_image[:, 2:3, ...]

        split_1 = patch_info_len // 3
        split_2 = 2 * patch_info_len // 3

        r_len = split_1
        g_len = split_2 - split_1
        b_len = patch_info_len - split_2

        r_recover_image, r_info = pee_extract(r_image, r_len)
        g_recover_image, g_info = pee_extract(g_image, g_len)
        b_recover_image, b_info = pee_extract(b_image, b_len)

        recover_image = torch.zeros_like(adv_image)
        recover_image[:, 0:1, ...] = r_recover_image
        recover_image[:, 1:2, ...] = g_recover_image
        recover_image[:, 2:3, ...] = b_recover_image

        info = []
        info.extend(r_info)
        info.extend(g_info)
        info.extend(b_info)

        if len(info) < patch_info_len:
            raise RuntimeError(
                f"Extracted info is too short: "
                f"expected {patch_info_len}, got {len(info)}"
            )

        info = info[:patch_info_len]

        mask_info = info[: self.mask_info_len]
        patch_info = info[self.mask_info_len :]

        mask = self.decode_mask(mask_info)

        patches = self.bin2patch(patch_info)
        patches = patches - self.attack_scale

        return recover_image.detach().clone(), patches, mask

    def encode_mask(self, mask):
        """
        Compress mask to 15 x 15 binary map.
        """
        mask = mask[:, 0:1, ...]

        mask = F.interpolate(
            mask,
            size=(15, 15),
            mode="nearest",
        )

        mask = mask.reshape(-1)
        mask = mask.tolist()
        mask = [int(x) for x in mask]

        return mask

    def decode_mask(self, compressed_mask):
        """
        Decode 15 x 15 mask to image_size x image_size.
        """
        if len(compressed_mask) < self.mask_info_len:
            raise RuntimeError(
                f"Mask info is too short: "
                f"expected {self.mask_info_len}, got {len(compressed_mask)}"
            )

        compressed_mask = compressed_mask[: self.mask_info_len]

        mask = torch.tensor(compressed_mask, dtype=torch.float32)
        mask = mask.reshape(1, 1, 15, 15)

        mask = F.interpolate(
            mask,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )

        mask = torch.cat([mask, mask, mask], dim=1)

        return mask


def tile_patch_to_image(patch: torch.Tensor, image_size: int, quant_step: int):
    """
    Tile a small perturbation patch to full image size.

    This function avoids shape mismatch when image_size is not divisible
    by block_size.

    patch:
        [B, C, ph, pw]

    return:
        [B, C, image_size, image_size]
    """
    _, _, ph, pw = patch.shape

    repeat_h = math.ceil(image_size / ph)
    repeat_w = math.ceil(image_size / pw)

    perturbation = patch.repeat(1, 1, repeat_h, repeat_w)
    perturbation = perturbation[..., :image_size, :image_size]
    perturbation = perturbation * quant_step

    return perturbation


def compute_predict_error(image: torch.Tensor):
    """
    Compute prediction error for PEE embedding/extraction.

    This version guarantees:
        predict_error.shape == image.shape

    image:
        [B, 1, H, W]

    return:
        [B, 1, H, W]
    """
    if image.dim() != 4:
        raise ValueError(f"Expected image shape [B, C, H, W], got {image.shape}")

    if image.size(1) != 1:
        raise ValueError(
            f"compute_predict_error expects single-channel image, "
            f"got {image.size(1)} channels"
        )

    device = image.device
    dtype = image.dtype

    _, _, h, w = image.shape

    kernel = torch.zeros(1, 1, 3, 3, device=device, dtype=dtype)
    kernel[..., 0, 1] = 0.25
    kernel[..., 1, 0] = 0.25
    kernel[..., 2, 1] = 0.25
    kernel[..., 1, 2] = 0.25

    center_kernel = torch.zeros_like(kernel)
    center_kernel[..., 1, 1] = 1

    predict_image_1 = torch.ceil(
        F.conv2d(
            image,
            kernel,
            stride=2,
            padding=0,
        )
    )

    grid_image_1 = F.conv2d(
        image,
        center_kernel,
        stride=2,
        padding=0,
    )

    predict_error_1 = grid_image_1 - predict_image_1

    predict_error_1 = F.conv_transpose2d(
        predict_error_1,
        center_kernel,
        stride=2,
    )

    inner_image = image[..., 1:-1, 1:-1]

    predict_image_2 = torch.ceil(
        F.conv2d(
            inner_image,
            kernel,
            stride=2,
            padding=0,
        )
    )

    grid_image_2 = F.conv2d(
        inner_image,
        center_kernel,
        stride=2,
        padding=0,
    )

    predict_error_2 = grid_image_2 - predict_image_2

    predict_error_2 = F.conv_transpose2d(
        predict_error_2,
        center_kernel,
        stride=2,
    )

    predict_error_2 = F.pad(
        predict_error_2,
        (1, 1, 1, 1),
        mode="constant",
        value=0,
    )

    predict_error = predict_error_1 + predict_error_2

    pe_h, pe_w = predict_error.shape[-2:]

    pad_h = h - pe_h
    pad_w = w - pe_w

    if pad_h > 0 or pad_w > 0:
        predict_error = F.pad(
            predict_error,
            (
                0, max(pad_w, 0),
                0, max(pad_h, 0),
            ),
            mode="constant",
            value=0,
        )

    predict_error = predict_error[..., :h, :w]

    if predict_error.shape != image.shape:
        raise RuntimeError(
            f"predict_error shape {predict_error.shape} "
            f"does not match image shape {image.shape}"
        )

    return predict_error


def pee_embed(image: torch.Tensor, info=None):
    """
    Prediction-error expansion embedding.

    This embedding rule is symmetric with pee_extract().

    Candidate locations:
        e == a or e == b

    Embed bit:
        bit = 0:
            e == a -> a
            e == b -> b

        bit = 1:
            e == a -> a - 1
            e == b -> b + 1

    Extraction rule:
        one_mask  = e == a - 1 or e == b + 1
        zero_mask = e == a     or e == b
    """
    if info is None:
        info = []

    with torch.no_grad():
        image = image.round().clamp(0, 255)
        predict_error = compute_predict_error(image)

        if predict_error.shape != image.shape:
            raise RuntimeError(
                f"predict_error shape {predict_error.shape} "
                f"does not match image shape {image.shape}"
            )

        flat_error = predict_error.flatten()
        unique_values, counts = torch.unique(flat_error, return_counts=True)

        max_bin = unique_values[counts.argmax()]
        a = max_bin - 1
        b = max_bin + 1

        embed_image = image.clone()

        # Histogram shifting to reserve a-1 and b+1.
        embed_image = torch.where(
            predict_error > b,
            embed_image + 1,
            embed_image,
        )

        embed_image = torch.where(
            predict_error < a,
            embed_image - 1,
            embed_image,
        )

        # Use both a and b bins for embedding.
        candidate_mask = (predict_error == a) | (predict_error == b)
        candidate_idx = candidate_mask.flatten().nonzero(as_tuple=False).flatten()

        capacity = candidate_idx.numel()

        if len(info) > capacity:
            raise RuntimeError(
                f"Embedding capacity is insufficient: "
                f"need {len(info)} bits, but only {capacity} embeddable positions. "
                f"Try increasing image_size, reducing block_size, reducing mask_num, "
                f"or reducing quant_step / bit_len."
            )

        if len(info) == 0:
            return embed_image.clamp(0, 255)

        info_tensor = torch.tensor(
            info,
            device=image.device,
            dtype=image.dtype,
        )

        flat_embed = embed_image.flatten()
        flat_predict_error = predict_error.flatten()

        selected_idx = candidate_idx[: len(info)]
        selected_error = flat_predict_error[selected_idx]

        a_one_mask = (selected_error == a) & (info_tensor == 1)
        b_one_mask = (selected_error == b) & (info_tensor == 1)

        # e == a and bit == 1: decrease pixel by 1, e becomes a - 1.
        flat_embed[selected_idx[a_one_mask]] -= 1

        # e == b and bit == 1: increase pixel by 1, e becomes b + 1.
        flat_embed[selected_idx[b_one_mask]] += 1

        embed_image = flat_embed.view_as(image)

        return embed_image.clamp(0, 255)


def pee_extract(image: torch.Tensor, info_len=0):
    """
    Prediction-error expansion extraction.

    Symmetric with pee_embed().
    """
    with torch.no_grad():
        image = image.round().clamp(0, 255)
        predict_error = compute_predict_error(image)

        if predict_error.shape != image.shape:
            raise RuntimeError(
                f"predict_error shape {predict_error.shape} "
                f"does not match image shape {image.shape}"
            )

        flat_error = predict_error.flatten()
        unique_values, counts = torch.unique(flat_error, return_counts=True)

        max_bin = unique_values[counts.argmax()]
        a = max_bin - 1
        b = max_bin + 1

        result_image = image.clone()

        one_mask = (predict_error == (a - 1)) | (predict_error == (b + 1))
        zero_mask = (predict_error == a) | (predict_error == b)
        bit_mask = one_mask | zero_mask

        bit_idx = bit_mask.flatten().nonzero(as_tuple=False).flatten()

        if bit_idx.numel() < info_len:
            raise RuntimeError(
                f"Extracted bit capacity is insufficient: "
                f"need {info_len} bits, but only {bit_idx.numel()} bits found."
            )

        bit_idx = bit_idx[:info_len]

        one_flat = one_mask.flatten()
        bits = one_flat[bit_idx].to(torch.int64).tolist()

        # Recover histogram shifting.
        result_image = torch.where(
            predict_error > b,
            result_image - 1,
            result_image,
        )

        result_image = torch.where(
            predict_error < a,
            result_image + 1,
            result_image,
        )

        return result_image.clamp(0, 255), bits


def scale_image(image, mask):
    """
    Scale masked region to leave room for perturbation.

    Original repository used:
        ((image - 127) // 2) + 127

    That maps 0 -> 63 and 255 -> 191, so adding -64 can underflow to -1.

    This version maps [0, 255] to [64, 191]:
        floor(image / 2) + 64

    Note:
        This operation discards the least significant bit.
        Strict pixel-wise reversible recovery would require saving that
        lost LSB information as auxiliary payload.
    """
    scaled = torch.floor(image / 2.0) + 64.0
    result_image = image * (1 - mask) + scaled * mask

    return result_image.detach().clone()


def recover_scale_image(scale_image, mask):
    """
    Approximate inverse of scale_image().

    Because scale_image() uses floor(image / 2), one LSB is lost unless
    auxiliary information is embedded and recovered.
    """
    image = (scale_image - 64.0) * 2.0
    image = image.clamp(0, 255)

    result_image = image * mask + scale_image * (1 - mask)

    return result_image.detach().clone()

def place_patch_by_mask(mask: torch.Tensor,
                        patch: torch.Tensor,
                        block_size: int,
                        quant_step: int):
    """
    Deterministically place non-overlapping patches inside mask.

    mask:
        [1, 3, H, W], binary or near-binary mask.

    patch:
        [1, 3, block_size, block_size]

    return:
        perturbation:
            [1, 3, H, W]
        placement_mask:
            [1, 3, H, W], actual patch placement region.
    """
    if mask.dim() != 4:
        raise ValueError(f"Expected mask shape [B, C, H, W], got {mask.shape}")

    if patch.dim() != 4:
        raise ValueError(f"Expected patch shape [B, C, h, w], got {patch.shape}")

    b, c, h, w = mask.shape
    _, _, ph, pw = patch.shape

    if ph != block_size or pw != block_size:
        raise ValueError(
            f"patch size must equal block_size, got patch={patch.shape}, "
            f"block_size={block_size}"
        )

    single_mask = mask[:, 0:1, :, :]
    perturbation = torch.zeros_like(mask)
    placement_mask = torch.zeros_like(mask)

    occupied = torch.zeros(
        (b, 1, h, w),
        device=mask.device,
        dtype=torch.bool,
    )

    num_patches = 0

    # Pixel-level scan: top-to-bottom, left-to-right.
    # This is more robust than scanning only on block_size grid,
    # because decoded masks may not align exactly with block_size.
    for y in range(0, h - block_size + 1):
        for x in range(0, w - block_size + 1):
            mask_region = single_mask[:, :, y:y + block_size, x:x + block_size]
            occ_region = occupied[:, :, y:y + block_size, x:x + block_size]

            # The whole patch must be inside mask and must not overlap.
            if mask_region.min().item() > 0.5 and not occ_region.any().item():
                perturbation[:, :, y:y + block_size, x:x + block_size] = (
                    patch * quant_step
                )

                placement_mask[:, :, y:y + block_size, x:x + block_size] = 1.0
                occupied[:, :, y:y + block_size, x:x + block_size] = True

                num_patches += 1

    if num_patches == 0:
        raise RuntimeError(
            "No valid non-overlapping patch can be placed inside the mask. "
            "Try increasing mask_num, increasing mask area, or reducing block_size."
        )

    return perturbation, placement_mask
