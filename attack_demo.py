import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from SRAP_Final import PEE_RAP_ATTACK


def load_image(image_path: str) -> torch.Tensor:
    """
    读取图片并转为 [C, H, W]，数值范围为 [0, 255]
    """
    image = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.ToTensor(),  # [0, 1]
    ])

    image_tensor = transform(image) * 255.0
    return image_tensor


def save_tensor_image(image_tensor: torch.Tensor, save_path: str):
    """
    保存图像 Tensor。
    支持 [B, C, H, W] 或 [C, H, W]，数值范围假设为 [0, 255]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image_tensor = image_tensor.detach().cpu()

    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]

    image_tensor = image_tensor.clamp(0, 255) / 255.0
    image = T.ToPILImage()(image_tensor)
    image.save(save_path)


def save_mask(mask: torch.Tensor, save_path: str):
    """
    保存 mask 可视化结果。
    mask: [B, C, H, W] 或 [C, H, W]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mask = mask.detach().cpu()

    if mask.dim() == 4:
        mask = mask[0]

    # 只保存单通道即可
    mask = mask[0:1].clamp(0, 1)
    mask_image = T.ToPILImage()(mask)
    mask_image.save(save_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--mask_num", type=int, default=8)
    parser.add_argument("--quant_step", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--cls_model", type=str, default="resnet50")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    attack_config = {
        "image_size": args.image_size,
        "block_size": args.block_size,
        "mask_num": args.mask_num,
        "quant_step": args.quant_step,
        "max_epoch": args.max_epoch,
        "device": args.device,
        "cls_model": args.cls_model,
    }

    attacker = PEE_RAP_ATTACK(attack_config)

    image = load_image(args.image_path)

    hide_adv_image, adv_label, mask = attacker.attack(
        image=image,
        label=args.label,
    )

    recover_image = attacker.recover(hide_adv_image)

    adv_save_path = os.path.join(args.output_dir, "hide_adv_image.png")
    recover_save_path = os.path.join(args.output_dir, "recover_image.png")
    mask_save_path = os.path.join(args.output_dir, "mask.png")

    save_tensor_image(hide_adv_image, adv_save_path)
    save_tensor_image(recover_image, recover_save_path)
    save_mask(mask, mask_save_path)

    print(f"Original label: {args.label}")
    print(f"Adversarial label: {adv_label.item()}")
    print(f"Hide adversarial image saved to: {adv_save_path}")
    print(f"Recovered image saved to: {recover_save_path}")
    print(f"Mask saved to: {mask_save_path}")


if __name__ == "__main__":
    main()
