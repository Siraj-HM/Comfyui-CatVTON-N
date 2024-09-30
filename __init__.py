import os
import torch
from .model.cloth_masker import AutoMasker as AM
from .model.cloth_masker import vis_mask
from .model.pipeline import CatVTONPipeline
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
import torchvision.transforms as T
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor

class LoadCatVTONPipeline:
    display_name = "Load CatVTON Pipeline"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd15_inpaint_path": ("STRING", {"default": "runwayml/stable-diffusion-inpainting"}),
                "catvton_path": ("STRING", {"default": "zhengchong/CatVTON"}),
                "mixed_precision": (["fp32", "fp16", "bf16"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "CatVTON" 
        
    def load(self, sd15_inpaint_path, catvton_path, mixed_precision):
        mixed_precision = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[mixed_precision]
        pipeline = CatVTONPipeline(
            base_ckpt=sd15_inpaint_path,
            attn_ckpt=catvton_path,
            attn_ckpt_version="mix",
            weight_dtype=mixed_precision,
            use_tf32=True,
            device='cuda'
        )
        return (pipeline,)

class LoadAutoMasker:
    display_name = "Load AutoMask Generator"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "catvton_path": ("STRING", {"default": "zhengchong/CatVTON"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "CatVTON" 
        
    def load(self, catvton_path):
        catvton_path = snapshot_download(repo_id=catvton_path)
        automasker = AM(
            densepose_ckpt=os.path.join(catvton_path, "DensePose"),
            schp_ckpt=os.path.join(catvton_path, "SCHP"),
            device='cuda', 
        )
        return (automasker,)

class CatVTON:
    display_name = "TryOn by CatVTON"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL",),
                "target_image": ("IMAGE",),
                "refer_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 2.5,
                        "min": 0.0,
                        "max": 14.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "CatVTON" 

    def generate(
        self, pipe: CatVTONPipeline, target_image, refer_image, mask_image, seed, steps, cfg
    ):
        target_image, refer_image, mask_image = [_.squeeze(0).permute(2, 0, 1) for _ in [target_image, refer_image, mask_image]]
        target_image = to_pil_image(target_image)
        refer_image = to_pil_image(refer_image)
        mask_image = mask_image[0]
        mask_image = to_pil_image(mask_image)
        generator = torch.Generator(device='cuda').manual_seed(seed)
        person_image = resize_and_crop(target_image, (768, 1024))
        cloth_image = resize_and_padding(refer_image, (768, 1024))
        mask = resize_and_crop(mask_image, (768, 1024))
        mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        mask = mask_processor.blur(mask, blur_factor=9)

        # Inference
        result_image = pipe(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator
        )[0]

        result_image = to_tensor(result_image).permute(1, 2, 0).unsqueeze(0)
        return (result_image,)

class AutoMasker:
    display_name = "Auto Mask Generation"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL",),
                "target_image": ("IMAGE",),
                "cloth_type": (["upper", "lower", 'overall'],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("mask", "masked_image")
    FUNCTION = "generate"
    CATEGORY = "CatVTON"

    def generate(self, pipe, target_image, cloth_type):
        print(f"Input target_image shape: {target_image.shape}")

        # Store original shape
        original_shape = target_image.shape

        # Ensure target_image is a PyTorch tensor
        if not isinstance(target_image, torch.Tensor):
            target_image = torch.from_numpy(target_image)

        # Remove batch dimension if present
        if target_image.dim() == 4:
            target_image = target_image.squeeze(0)

        # Ensure the tensor is in the correct format (C, H, W)
        if target_image.shape[-1] == 3:
            target_image = target_image.permute(2, 0, 1)

        # Store original size
        original_height, original_width = target_image.shape[1:]

        # Normalize to 0-1 range if necessary
        if target_image.max() > 1.0:
            target_image = target_image.float() / 255.0

        # Log input size
        print(f"Processing image of size: {target_image.shape}")

        # Resize target image for processing (keep the original size)
        resize_transform = T.Resize((original_height, original_width), antialias=True)
        processed_image = resize_transform(target_image)

        # Convert to PIL Image for pipe input
        pil_image = T.ToPILImage()(processed_image)

        # Generate mask
        try:
            result = pipe(pil_image, cloth_type)
            mask = result['mask']
        except Exception as e:
            print(f"Error in mask generation: {e}")
            # Return black images if mask generation fails
            black_image = torch.zeros((original_height, original_width, 3), dtype=torch.uint8)
            return (black_image.unsqueeze(0), black_image.unsqueeze(0))

        # Ensure mask is a PyTorch tensor
        if isinstance(mask, Image.Image):
            mask = T.ToTensor()(mask)
        elif not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()

        # Resize the mask to match the original size of the target image
        mask = T.Resize((original_height, original_width), antialias=False)(mask)

        # Ensure mask is binary and 3-channel
        mask = (mask > 0.5).float()
        if mask.shape[0] == 1:
            mask = mask.repeat(3, 1, 1)

        # Create masked image
        masked_image = target_image * mask + target_image * (1 - mask)

        # Ensure output is in the expected range (0-255) and format (H, W, C)
        mask = (mask.permute(1, 2, 0) * 255).byte()
        masked_image = (masked_image.permute(1, 2, 0) * 255).byte()

        # Re-add batch dimension
        mask = mask.unsqueeze(0)
        masked_image = masked_image.unsqueeze(0)

        print(f"Output mask shape: {mask.shape}")
        print(f"Output masked_image shape: {masked_image.shape}")

        return (mask, masked_image)



_export_classes = [
    LoadCatVTONPipeline,
    LoadAutoMasker,
    CatVTON,
    AutoMasker,
]

NODE_CLASS_MAPPINGS = {c.__name__: c for c in _export_classes}

NODE_DISPLAY_NAME_MAPPINGS = {
    c.__name__: getattr(c, "display_name", c.__name__) for c in _export_classes
}

print(f"AutoMasker RETURN_TYPES: {AutoMasker.RETURN_TYPES}")
print(f"AutoMasker RETURN_NAMES: {AutoMasker.RETURN_NAMES}")
