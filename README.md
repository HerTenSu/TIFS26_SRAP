# IEEE Transactions on Information Forensics and Security
# SRAP: Robust and Transferable Self-Reversible Adversarial Patch for Image Privacy Protection
````markdown
# SRAP Attack Demo

This repository provides a PyTorch implementation example for running the SRAP-style self-reversible adversarial patch attack on an input image.

## Requirements

```bash
pip install torch torchvision pillow
````

## Usage

Run the demo script with an input image and its ImageNet label:

```bash
python attack_demo.py \
  --image_path ./test.JPEG \
  --label 999 \
  --output_dir ./outputs \
  --cls_model resnet50 \
  --image_size 225 \
  --block_size 5 \
  --mask_num 256 \
  --quant_step 4 \
  --max_epoch 20 \
  --device cuda
```


## Arguments

| Argument       | Description                                            |
| -------------- | ------------------------------------------------------ |
| `--image_path` | Path to the input image.                               |
| `--label`      | ImageNet class index of the input image.               |
| `--output_dir` | Directory for saving results.                          |
| `--cls_model`  | Target classification model from `torchvision.models`. |
| `--image_size` | Input image size used by the attack.                   |
| `--block_size` | Patch block size.                                      |
| `--mask_num`   | Number of selected attack blocks.                      |
| `--quant_step` | Quantization step for the perturbation.                |
| `--max_epoch`  | Maximum number of attack iterations.                   |
| `--device`     | Runtime device, e.g., `cuda` or `cpu`.                 |

## Outputs

After running the script, the output directory contains:

```text
outputs/
├── hide_adv_image.png
├── recover_image.png
└── mask.png
```

| File                 | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `hide_adv_image.png` | Adversarial image with embedded recovery information. |
| `recover_image.png`  | Recovered image extracted from the adversarial image. |
| `mask.png`           | Visualization of the selected attack region.          |

## Citation

If you use this method or code, please cite:

```bibtex
@ARTICLE{11450347, 
  author={Zhao, Zeyu and Xu, Ke and Sun, Tanfeng and Jiang, Xinghao},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={SRAP: Robust and Transferable Self-Reversible Adversarial Patch for Image Privacy Protection}, 
  year={2026},
  volume={21},
  number={},
  pages={3689-3702},
  keywords={Artificial intelligence;Perturbation methods;Robustness;Protection;Privacy;Mathematical models;Image restoration;Data mining;Noise;Analytical models;Reversible adversarial examples;adversarial patch;image protection;adversarial transferability;robust adversarial attack},
  doi={10.1109/TIFS.2026.3676667}
}
```

