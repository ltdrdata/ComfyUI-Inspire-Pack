import torch
from . import a1111_compat
import comfy
from .libs import common, utils
from comfy import model_management
from comfy_extras import nodes_custom_sampler
import nodes


class KSampler_progress(a1111_compat.KSampler_inspire):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (common.SCHEDULERS, ),
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

    @staticmethod
    def doit(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, noise_mode, interval, omit_start_latent):
        adv_steps = int(steps / denoise)

        if omit_start_latent:
            result = []
        else:
            result = [latent_image['samples']]

        result = []

        def progress_callback(step, x0, x, total_steps):
            if (total_steps-1) != step and step % interval != 0:
                return

            x = model.model.process_latent_out(x)
            x = x.to(model_management.intermediate_device())
            result.append(x)

        latent_image, noise = a1111_compat.KSamplerAdvanced_inspire.sample(model, True, seed, adv_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, (adv_steps-steps), adv_steps, noise_mode, False, callback=progress_callback)

        if len(result) > 0:
            result = torch.cat(result)
            result = {'samples': result}
        else:
            result = latent_image

        return latent_image, result


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
                     "scheduler": (common.SCHEDULERS, ),
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

    FUNCTION = "doit"

    CATEGORY = "InspirePack/analysis"

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "progress_latent")

    def doit(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step,
             noise_mode, return_with_leftover_noise, interval, omit_start_latent, prev_progress_latent_opt=None):
        if omit_start_latent:
            result = []
        else:
            result = [latent_image['samples']]

        result = []

        def progress_callback(step, x0, x, total_steps):
            if (total_steps-1) != step and step % interval != 0:
                return

            x = model.model.process_latent_out(x)
            x = x.to(model_management.intermediate_device())
            result.append(x)

        latent_image, noise = a1111_compat.KSamplerAdvanced_inspire.sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step,
                                                                           noise_mode, False, callback=progress_callback)

        if len(result) > 0:
            result = torch.cat(result)
            result = {'samples': result}
        else:
            result = latent_image

        if prev_progress_latent_opt is not None:
            result['samples'] = torch.cat((prev_progress_latent_opt['samples'], result['samples']), dim=0)

        return latent_image, result


NODE_CLASS_MAPPINGS = {
    "KSamplerProgress //Inspire": KSampler_progress,
    "KSamplerAdvancedProgress //Inspire": KSamplerAdvanced_progress,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerProgress //Inspire": "KSampler Progress (Inspire)",
    "KSamplerAdvancedProgress //Inspire": "KSampler Advanced Progress (Inspire)",
}
