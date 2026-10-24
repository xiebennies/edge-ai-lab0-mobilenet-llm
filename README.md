# Edge AI Lab 0: Vision Model & SLM on GPU

Two parts:
1. Fine-tune MobileNetV2 (ImageNet pretrained) on CIFAR-10
2. Speed up Llama-3.2-1B-Instruct inference with `torch.compile`

## Results
- CIFAR-10 test accuracy: 92.8%
- Inference speedup with torch.compile: ~1.51x (median time 1.69s -> 1.12s)

## Environment
- RTX 3060 Laptop GPU (6GB), WSL2 Ubuntu 22.04
- PyTorch 2.6 + CUDA 12.4, transformers 4.44.2

## Notes
- Static KV cache + compiling `model.forward` (not the whole model) was needed to make `torch.compile` work with `generate()`.
- Newer transformers versions broke compile in this setup; pinning 4.44.2 fixed it.
