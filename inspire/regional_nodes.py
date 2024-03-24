import traceback

import comfy
import nodes
import torch
from . import prompt_support
from .libs import utils


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
                "controlnet_in_pipe": ("BOOLEAN", {"default": False, "label_on": "Keep", "label_off": "Override"}),
                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt, controlnet_in_pipe=False, sigma_factor=1.0):
        if 'RegionalPrompt' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/ltdrdata/ComfyUI-Impact-Pack',
                                          "To use 'RegionalPromptSimple' node, 'Impact Pack' extension is required.")
            raise Exception(f"[ERROR] To use RegionalPromptSimple, you need to install 'ComfyUI-Impact-Pack'")

        model, clip, vae, positive, negative = basic_pipe

        iwe = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardEncode']()
        kap = nodes.NODE_CLASS_MAPPINGS['KSamplerAdvancedProvider']()
        rp = nodes.NODE_CLASS_MAPPINGS['RegionalPrompt']()

        if wildcard_prompt != "":
            model, clip, new_positive, _ = iwe.doit(model=model, clip=clip, populated_text=wildcard_prompt)

            if controlnet_in_pipe:
                prev_cnet = None
                for t in positive:
                    if 'control' in t[1] and 'control_apply_to_uncond' in t[1]:
                        prev_cnet = t[1]['control'], t[1]['control_apply_to_uncond']
                        break

                if prev_cnet is not None:
                    for t in new_positive:
                        t[1]['control'] = prev_cnet[0]
                        t[1]['control_apply_to_uncond'] = prev_cnet[1]

        else:
            new_positive = positive

        basic_pipe = model, clip, vae, new_positive, negative

        sampler = kap.doit(cfg, sampler_name, scheduler, basic_pipe, sigma_factor=sigma_factor)[0]
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
                "controlnet_in_pipe": ("BOOLEAN", {"default": False, "label_on": "Keep", "label_off": "Override"}),
                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("REGIONAL_PROMPTS", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, basic_pipe, color_mask, mask_color, cfg, sampler_name, scheduler, wildcard_prompt, controlnet_in_pipe=False, sigma_factor=1.0):
        mask = color_to_mask(color_mask, mask_color)
        rp = RegionalPromptSimple().doit(basic_pipe, mask, cfg, sampler_name, scheduler, wildcard_prompt, controlnet_in_pipe, sigma_factor=sigma_factor)[0]
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
                "model": ("MODEL",),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            }
        }

    RETURN_TYPES = ("IPADAPTER_PIPE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, ipadapter, model, clip_vision, insightface=None):
        pipe = ipadapter, model, clip_vision, insightface, lambda x: x

        return (pipe,)


class FromIPAdapterPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter_pipe": ("IPADAPTER_PIPE", ),
            }
        }

    RETURN_TYPES = ("IPADAPTER", "MODEL", "CLIP_VISION", "INSIGHTFACE")
    RETURN_NAMES = ("ipadapter", "model", "clip_vision", "insight_face")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, ipadapter_pipe):
        ipadapter, model, clip_vision, insightface, _ = ipadapter_pipe
        return ipadapter, model, clip_vision, insightface


class IPAdapterConditioning:
    def __init__(self, mask, weight, weight_type, noise=None, image=None, neg_image=None, embeds=None, start_at=0.0, end_at=1.0, combine_embeds='concat', unfold_batch=False, weight_v2=False, neg_embeds=None):
        self.mask = mask
        self.image = image
        self.neg_image = neg_image
        self.embeds = embeds
        self.neg_embeds = neg_embeds
        self.weight = weight
        self.noise = noise
        self.weight_type = weight_type
        self.start_at = start_at
        self.end_at = end_at
        self.unfold_batch = unfold_batch
        self.weight_v2 = weight_v2
        self.combine_embeds = combine_embeds

    def doit(self, ipadapter_pipe):
        ipadapter, model, clip_vision, insightface, _ = ipadapter_pipe

        if 'IPAdapterAdvanced' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/cubiq/ComfyUI_IPAdapter_plus',
                                          "To use 'Regional IPAdapter' node, 'ComfyUI IPAdapter Plus' extension is required.")
            raise Exception(f"[ERROR] To use IPAdapterModelHelper, you need to install 'ComfyUI IPAdapter Plus'")

        if self.embeds is None:
            obj = nodes.NODE_CLASS_MAPPINGS['IPAdapterAdvanced']
            model = obj().apply_ipadapter(model=model, ipadapter=ipadapter, weight=self.weight, weight_type=self.weight_type,
                                          start_at=self.start_at, end_at=self.end_at, combine_embeds=self.combine_embeds,
                                          clip_vision=clip_vision, image=self.image, image_negative=self.neg_image, attn_mask=self.mask,
                                          insightface=insightface, weight_faceidv2=self.weight_v2)[0]
        else:
            obj = nodes.NODE_CLASS_MAPPINGS['IPAdapterEmbeds']
            model = obj().apply_ipadapter(model=model, ipadapter=ipadapter, pos_embed=self.embeds, weight=self.weight, weight_type=self.weight_type,
                                          start_at=self.start_at, end_at=self.end_at, neg_embed=self.neg_embeds,
                                          attn_mask=self.mask, clip_vision=clip_vision)[0]

        return model


class RegionalIPAdapterMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),

                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (["original", "linear", "channel penalty"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "faceid_v2": ("BOOLEAN", {"default": False}),
                "weight_v2": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "neg_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, mask, image, weight, noise, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, faceid_v2=False, weight_v2=False, combine_embeds="concat", neg_image=None):
        cond = IPAdapterConditioning(mask, weight, weight_type, noise=noise, image=image, neg_image=neg_image, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, weight_v2=weight_v2, combine_embeds=combine_embeds)
        return (cond, )


class RegionalIPAdapterColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (["original", "linear", "channel penalty"], ),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "faceid_v2": ("BOOLEAN", {"default": False }),
                "weight_v2": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "neg_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, color_mask, mask_color, image, weight, noise, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, faceid_v2=False, weight_v2=False, combine_embeds="concat", neg_image=None):
        mask = color_to_mask(color_mask, mask_color)
        cond = IPAdapterConditioning(mask, weight, weight_type, noise=noise, image=image, neg_image=neg_image, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, weight_v2=weight_v2, combine_embeds=combine_embeds)
        return (cond, mask)


class RegionalIPAdapterEncodedMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),

                "embeds": ("EMBEDS",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "weight_type": (["original", "linear", "channel penalty"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "neg_embeds": ("EMBEDS",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, mask, embeds, weight, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, neg_embeds=None):
        cond = IPAdapterConditioning(mask, weight, weight_type, embeds=embeds, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, neg_embeds=neg_embeds)
        return (cond, )


class RegionalIPAdapterEncodedColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),

                "embeds": ("EMBEDS",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "weight_type": (["original", "linear", "channel penalty"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "neg_embeds": ("EMBEDS",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, color_mask, mask_color, embeds, weight, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, neg_embeds=None):
        mask = color_to_mask(color_mask, mask_color)
        cond = IPAdapterConditioning(mask, weight, weight_type, embeds=embeds, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, neg_embeds=neg_embeds)
        return (cond, mask)


class ApplyRegionalIPAdapters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "ipadapter_pipe": ("IPADAPTER_PIPE",),
                    "regional_ipadapter1": ("REGIONAL_IPADAPTER", ),
                    },
                }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, **kwargs):
        ipadapter_pipe = kwargs['ipadapter_pipe']
        ipadapter, model, clip_vision, insightface, lora_loader = ipadapter_pipe

        del kwargs['ipadapter_pipe']

        for k, v in kwargs.items():
            ipadapter_pipe = ipadapter, model, clip_vision, insightface, lora_loader
            model = v.doit(ipadapter_pipe)

        return (model, )


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
    "RegionalIPAdapterEncodedMask //Inspire": RegionalIPAdapterEncodedMask,
    "RegionalIPAdapterEncodedColorMask //Inspire": RegionalIPAdapterEncodedColorMask,
    "RegionalSeedExplorerMask //Inspire": RegionalSeedExplorerMask,
    "RegionalSeedExplorerColorMask //Inspire": RegionalSeedExplorerColorMask,
    "ToIPAdapterPipe //Inspire": ToIPAdapterPipe,
    "FromIPAdapterPipe //Inspire": FromIPAdapterPipe,
    "ApplyRegionalIPAdapters //Inspire": ApplyRegionalIPAdapters,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionalPromptSimple //Inspire": "Regional Prompt Simple (Inspire)",
    "RegionalPromptColorMask //Inspire": "Regional Prompt By Color Mask (Inspire)",
    "RegionalConditioningSimple //Inspire": "Regional Conditioning Simple (Inspire)",
    "RegionalConditioningColorMask //Inspire": "Regional Conditioning By Color Mask (Inspire)",
    "RegionalIPAdapterMask //Inspire": "Regional IPAdapter Mask (Inspire)",
    "RegionalIPAdapterColorMask //Inspire": "Regional IPAdapter By Color Mask (Inspire)",
    "RegionalIPAdapterEncodedMask //Inspire": "Regional IPAdapter Encoded Mask (Inspire)",
    "RegionalIPAdapterEncodedColorMask //Inspire": "Regional IPAdapter Encoded By Color Mask (Inspire)",
    "RegionalSeedExplorerMask //Inspire": "Regional Seed Explorer By Mask (Inspire)",
    "RegionalSeedExplorerColorMask //Inspire": "Regional Seed Explorer By Color Mask (Inspire)",
    "ToIPAdapterPipe //Inspire": "ToIPAdapterPipe (Inspire)",
    "FromIPAdapterPipe //Inspire": "FromIPAdapterPipe (Inspire)",
    "ApplyRegionalIPAdapters //Inspire": "Apply Regional IPAdapters (Inspire)"
}
