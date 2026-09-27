import torch


def gpu_dtype(device: str) -> torch.dtype:
    """bf16 on GPUs that support it: fp16 overflowed to NaN in parakeet-ctc-0.6b; bf16 has fp32's range. fp32 on CPU."""
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32
