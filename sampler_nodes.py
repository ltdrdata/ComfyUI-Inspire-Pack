import torch
from . import a1111_compat
import comfy

class KSampler_progress(a1111_compat.KSampler_inspire):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                     "positive": ("CONDITIONING", ),
                     "negative": ("CONDITIONING", ),
                     "latent_image": ("LATENT", ),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "noise_mode": (["GPU(=A1111)", "CPU"],),
                     "interval": ("INT", {"default": 1, "min": 1, "max": 10000}),
                     "omit_start_latent": ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
                     }
                }

    CATEGORY = "InspirePack/analysis"

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "progress_latent")

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, noise_mode, interval, omit_start_latent):
        adv_steps = int(steps / denoise)

        sampler = a1111_compat.KSamplerAdvanced_inspire()

        if omit_start_latent:
            result = []
        else:
            result = [latent_image['samples']]

        for i in range(0, adv_steps+1):
            add_noise = i == 0
            return_with_leftover_noise = i != adv_steps
            latent_image = sampler.sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, i, i+1, noise_mode, return_with_leftover_noise)[0]
            if i % interval == 0 or i == adv_steps:
                result.append(latent_image['samples'])

        if len(result) > 0:
            result = torch.cat(result)
            result = {'samples': result}
        else:
            result = latent_image

        return (latent_image, result)


class KSamplerAdvanced_progress(a1111_compat.KSamplerAdvanced_inspire):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                     "positive": ("CONDITIONING", ),
                     "negative": ("CONDITIONING", ),
                     "latent_image": ("LATENT", ),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "noise_mode": (["GPU(=A1111)", "CPU"],),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                     "interval": ("INT", {"default": 1, "min": 1, "max": 10000}),
                     "omit_start_latent": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                     },
                "optional": {"prev_progress_latent_opt": ("LATENT",), }
                }

    FUNCTION = "sample"

    CATEGORY = "InspirePack/analysis"

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "progress_latent")

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, noise_mode, return_with_leftover_noise, interval, omit_start_latent, prev_progress_latent_opt=None):
        sampler = a1111_compat.KSamplerAdvanced_inspire()

        if omit_start_latent:
            result = []
        else:
            result = [latent_image['samples']]

        for i in range(start_at_step, end_at_step+1):
            cur_add_noise = i == start_at_step and add_noise
            cur_return_with_leftover_noise = i != steps or return_with_leftover_noise
            latent_image = sampler.sample(model, cur_add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, i, i+1, noise_mode, cur_return_with_leftover_noise)[0]
            print(f"{i}, {i+1}")
            if i % interval == 0 or i == steps:
                result.append(latent_image['samples'])

        if len(result) > 0:
            result = torch.cat(result)
            result = {'samples': result}
        else:
            result = latent_image

        if prev_progress_latent_opt is not None:
            result['samples'] = torch.cat((prev_progress_latent_opt['samples'], result['samples']), dim=0)

        return (latent_image, result)


NODE_CLASS_MAPPINGS = {
    "KSamplerProgress //Inspire": KSampler_progress,
    "KSamplerAdvancedProgress //Inspire": KSamplerAdvanced_progress,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerProgress //Inspire": "KSampler Progress (Inspire)",
    "KSamplerAdvancedProgress //Inspire": "KSampler Advanced Progress (Inspire)",
}
