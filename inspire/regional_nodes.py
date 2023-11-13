import traceback

import comfy
import nodes
import numpy as np
import torch
import torch.nn.functional as F
from . import prompt_support

class RegionalPromptSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_pipe": ("BASIC_PIPE",),
                "mask": ("MASK",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "wildcard_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "wildcard prompt"}),
            },
        }

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt):
        if 'RegionalPrompt' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use RegionalPromptSimple, you need to install 'ComfyUI-Impact-Pack'")

        model, clip, vae, positive, negative = basic_pipe

        iwe = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardEncode']()
        kap = nodes.NODE_CLASS_MAPPINGS['KSamplerAdvancedProvider']()
        rp = nodes.NODE_CLASS_MAPPINGS['RegionalPrompt']()

        if wildcard_prompt != "":
            model, clip, positive, _ = iwe.doit(model=model, clip=clip, populated_text=wildcard_prompt)

        basic_pipe = model, clip, vae, positive, negative

        sampler = kap.doit(cfg, sampler_name, scheduler, basic_pipe)[0]
        regional_prompts = rp.doit(mask, sampler)[0]

        return (regional_prompts, )


def color_to_mask(color_mask, mask_color):
    try:
        if mask_color.startswith("#"):
            selected = int(mask_color[1:], 16)
        else:
            selected = int(mask_color, 10)
    except Exception:
        raise Exception(f"[ERROR] Invalid mask_color value. mask_color should be a color value for RGB")

    temp = (torch.clamp(color_mask, 0, 1.0) * 255.0).round().to(torch.int)
    temp = torch.bitwise_left_shift(temp[:, :, :, 0], 16) + torch.bitwise_left_shift(temp[:, :, :, 1], 8) + temp[:, :, :, 2]
    mask = torch.where(temp == selected, 1.0, 0.0)
    return mask


class RegionalPromptColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basic_pipe": ("BASIC_PIPE",),
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "wildcard_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "wildcard prompt"}),
            },
        }

    RETURN_TYPES = ("REGIONAL_PROMPTS", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, basic_pipe, color_mask, mask_color, cfg, sampler_name, scheduler, wildcard_prompt):
        mask = color_to_mask(color_mask, mask_color)
        rp = RegionalPromptSimple().doit(basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt)[0]
        return (rp, mask)


class RegionalConditioningSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "mask": ("MASK",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
                "prompt": ("STRING", {"multiline": True, "placeholder": "prompt"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, clip, mask, strength, set_cond_area, prompt):
        conditioning = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        conditioning = nodes.ConditioningSetMask().append(conditioning, mask, set_cond_area, strength)[0]
        return (conditioning, )


class RegionalConditioningColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
                "prompt": ("STRING", {"multiline": True, "placeholder": "prompt"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, clip, color_mask, mask_color, strength, set_cond_area, prompt):
        mask = color_to_mask(color_mask, mask_color)

        conditioning = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        conditioning = nodes.ConditioningSetMask().append(conditioning, mask, set_cond_area, strength)[0]
        return (conditioning, mask)


class ToIPAdapterPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "clip_vision": ("CLIP_VISION",),
                "model": ("MODEL", ),
            }
        }

    RETURN_TYPES = ("IPADAPTER_PIPE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, ipadapter, clip_vision, model):
        pipe = ipadapter, clip_vision, model

        return (pipe,)


class FromIPAdapterPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_pipe": ("IPADAPTER_PIPE", ),
            }
        }

    RETURN_TYPES = ("IPADAPTER", "CLIP_VISION", "MODEL")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, ipadapter_pipe):
        return ipadapter_pipe


class RegionalIPAdapterMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_pipe": ("IPADAPTER_PIPE",),
                "mask": ("MASK",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (["original", "linear", "channel penalty"],),
            },
        }

    RETURN_TYPES = ("IPADAPTER_PIPE", "MODEL")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, ipadapter_pipe, mask, image, weight, noise, weight_type):
        if 'IPAdapterApply' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use RegionalIPAdapterMask, you need to install 'ComfyUI_IPAdapter_plus'")

        obj = nodes.NODE_CLASS_MAPPINGS['IPAdapterApply']

        ipadapter, clip_vision, model = ipadapter_pipe
        model = obj().apply_ipadapter(ipadapter, model, weight, clip_vision=clip_vision, image=image, weight_type=weight_type, noise=noise, attn_mask=mask)[0]

        new_ipadapter_pipe = ipadapter, clip_vision, model
        return new_ipadapter_pipe, model


class RegionalIPAdapterColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_pipe": ("IPADAPTER_PIPE", ),

                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (["original", "linear", "channel penalty"], ),
            },
        }

    RETURN_TYPES = ("IPADAPTER_PIPE", "MODEL", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, ipadapter_pipe, color_mask, mask_color, image, weight, noise, weight_type):
        mask = color_to_mask(color_mask, mask_color)

        if 'IPAdapterApply' not in nodes.NODE_CLASS_MAPPINGS:
            raise Exception(f"[ERROR] To use RegionalIPAdapterColorMask, you need to install 'ComfyUI_IPAdapter_plus'")
        
        obj = nodes.NODE_CLASS_MAPPINGS['IPAdapterApply']

        ipadapter, clip_vision, model = ipadapter_pipe
        model = obj().apply_ipadapter(ipadapter, model, weight, clip_vision=clip_vision, image=image, weight_type=weight_type, noise=noise, attn_mask=mask)[0]

        new_ipadapter_pipe = ipadapter, clip_vision, model
        return new_ipadapter_pipe, model, mask


class RegionalSeedExplorerMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),

                "noise": ("NOISE",),
                "seed_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": False}),
                "enable_additional": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "additional_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "additional_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_mode": (["GPU(=A1111)", "CPU"],),
            },
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, mask, noise, seed_prompt, enable_additional, additional_seed, additional_strength, noise_mode):
        device = comfy.model_management.get_torch_device()
        noise_device = "cpu" if noise_mode == "CPU" else device

        noise = noise.to(device)
        mask = mask.to(device)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(noise.shape[2], noise.shape[3]), mode="bilinear").squeeze(0)

        try:
            seed_prompt = seed_prompt.replace("\n", "")
            items = seed_prompt.strip().split(",")

            if items == ['']:
                items = []

            if enable_additional:
                items.append((additional_seed, additional_strength))

            noise = prompt_support.SeedExplorer.apply_variation(noise, items, noise_device, mask)
        except Exception:
            print(f"[ERROR] IGNORED: RegionalSeedExplorerColorMask is failed.")
            traceback.print_exc()

        noise = noise.cpu()
        mask.cpu()
        return (noise,)


class RegionalSeedExplorerColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),

                "noise": ("NOISE",),
                "seed_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": False}),
                "enable_additional": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "additional_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "additional_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_mode": (["GPU(=A1111)", "CPU"],),
            },
        }

    RETURN_TYPES = ("NOISE", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, color_mask, mask_color, noise, seed_prompt, enable_additional, additional_seed, additional_strength, noise_mode):
        device = comfy.model_management.get_torch_device()
        noise_device = "cpu" if noise_mode == "CPU" else device

        color_mask = color_mask.to(device)
        noise = noise.to(device)

        mask = color_to_mask(color_mask, mask_color)
        original_mask = mask
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(noise.shape[2], noise.shape[3]), mode="bilinear").squeeze(0)

        mask = mask.to(device)

        try:
            seed_prompt = seed_prompt.replace("\n", "")
            items = seed_prompt.strip().split(",")

            if items == ['']:
                items = []

            if enable_additional:
                items.append((additional_seed, additional_strength))

            noise = prompt_support.SeedExplorer.apply_variation(noise, items, noise_device, mask)
        except Exception:
            print(f"[ERROR] IGNORED: RegionalSeedExplorerColorMask is failed.")
            traceback.print_exc()

        color_mask.cpu()
        noise = noise.cpu()
        original_mask = original_mask.cpu()
        return (noise, original_mask)


NODE_CLASS_MAPPINGS = {
    "RegionalPromptSimple //Inspire": RegionalPromptSimple,
    "RegionalPromptColorMask //Inspire": RegionalPromptColorMask,
    "RegionalConditioningSimple //Inspire": RegionalConditioningSimple,
    "RegionalConditioningColorMask //Inspire": RegionalConditioningColorMask,
    "RegionalIPAdapterMask //Inspire": RegionalIPAdapterMask,
    "RegionalIPAdapterColorMask //Inspire": RegionalIPAdapterColorMask,
    "RegionalSeedExplorerMask //Inspire": RegionalSeedExplorerMask,
    "RegionalSeedExplorerColorMask //Inspire": RegionalSeedExplorerColorMask,
    "ToIPAdapterPipe //Inspire": ToIPAdapterPipe,
    "FromIPAdapterPipe //Inspire": FromIPAdapterPipe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionalPromptSimple //Inspire": "Regional Prompt Simple (Inspire)",
    "RegionalPromptColorMask //Inspire": "Regional Prompt By Color Mask (Inspire)",
    "RegionalConditioningSimple //Inspire": "Regional Conditioning Simple (Inspire)",
    "RegionalConditioningColorMask //Inspire": "Regional Conditioning By Color Mask (Inspire)",
    "RegionalIPAdapterMask //Inspire": "Regional IPAdapter Mask (Inspire)",
    "RegionalIPAdapterColorMask //Inspire": "Regional IPAdapter By Color Mask (Inspire)",
    "RegionalSeedExplorerMask //Inspire": "Regional Seed Explorer By Mask (Inspire)",
    "RegionalSeedExplorerColorMask //Inspire": "Regional Seed Explorer By Color Mask (Inspire)",
    "ToIPAdapterPipe //Inspire": "ToIPAdapterPipe (Inspire)",
    "FromIPAdapterPipe //Inspire": "FromIPAdapterPipe (Inspire)",
}
