"""Native SigLIP-base features; torch is imported only when loading a model."""
import re
import time

MODEL_ID = "google/siglip-base-patch16-224"


def resolve_revision(revision, allow_download=False):
    """Resolve an online branch once, then load processor and weights at one SHA."""
    if revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    if not allow_download:
        raise ValueError("Offline loading requires --revision with the 40-character SHA from a previous run")
    from huggingface_hub import HfApi
    sha = HfApi().model_info(MODEL_ID, revision=revision or "main", timeout=30,
                             token=False).sha
    if not sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("The Hub did not return an immutable model revision")
    return sha


class SiglipEncoder:
    """Load on CPU first; callers can use features with or without gradients."""

    def __init__(self, revision=None, *, allow_download=False, device="cuda:0",
                 precision="float32"):
        if device not in ("cpu", "cuda:0"):
            raise ValueError("Use cpu or the isolated GPU's logical cuda:0")
        if precision not in ("float32", "bfloat16"):
            raise ValueError("Unsupported precision")
        if device == "cpu" and precision != "float32":
            raise ValueError("The CPU smoke test uses float32")
        self.revision = resolve_revision(revision, allow_download)
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device
        self.dtype = getattr(torch, precision)
        common = {"revision": self.revision, "local_files_only": not allow_download,
                  "trust_remote_code": False, "token": False}
        start = time.perf_counter()
        print(f"Loading {MODEL_ID}@{self.revision} on CPU...", flush=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False, **common)
        self.model = AutoModel.from_pretrained(MODEL_ID, dtype=self.dtype,
                                               use_safetensors=True,
                                               attn_implementation="sdpa", **common)
        config = self.model.config
        if (config.model_type != "siglip" or config.vision_config.image_size != 224
                or config.vision_config.patch_size != 16
                or config.text_config.max_position_embeddings != 64):
            raise ValueError("Loaded configuration does not match SigLIP-base-patch16-224")
        self.load_seconds = time.perf_counter() - start
        if device == "cuda:0":
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError("This smoke test requires exactly one explicitly selected visible GPU")
            if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("The selected GPU does not support bfloat16")
        print(f"Files ready; moving model to {device} for the smoke test...", flush=True)
        self.model.to(device).eval()

    def features(self, images, texts):
        if not images or not texts or any(not text.strip() for text in texts):
            raise ValueError("Images and nonempty captions are required")
        image_inputs = self.processor(images=images, return_tensors="pt")
        text_inputs = self.processor(text=[text.lower() for text in texts],
                                     padding="max_length", max_length=64,
                                     truncation=True, return_tensors="pt")
        shapes = {"pixel_values": list(image_inputs["pixel_values"].shape),
                  "input_ids": list(text_inputs["input_ids"].shape)}
        pixels = image_inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()
                       if key in ("input_ids", "attention_mask")}
        image_features = self.model.get_image_features(pixel_values=pixels)
        text_features = self.model.get_text_features(**text_inputs)

        def normalize(features):
            features = features.float()
            norms = features.norm(dim=-1, keepdim=True)
            if not self.torch.isfinite(features).all() or not (norms > 0).all():
                raise ValueError("Nonfinite or zero model features")
            return features / norms

        return normalize(image_features), normalize(text_features), shapes
