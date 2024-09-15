import torch
from . import a1111_compat
import comfy
from .libs import common
from comfy import model_management
from comfy.samplers import CFGGuider
from comfy_extras.nodes_perpneg import Guider_PerpNeg
import math

class KSampler_progress(a1111_compat.KSampler_inspire):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
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
                    "omit_final_latent": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                    },
                "optional": {
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    CATEGORY = "InspirePack/analysis"

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "progress_latent")

    @staticmethod
    def doit(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, noise_mode,
             interval, omit_start_latent, omit_final_latent, scheduler_func_opt=None):
        adv_steps = int(steps / denoise)

        if omit_start_latent:
            result = []
        else:
            result = [comfy.sample.fix_empty_latent_channels(model, latent_image['samples']).cpu()]

        def progress_callback(step, x0, x, total_steps):
            if (total_steps-1) != step and step % interval != 0:
                return

            x = model.model.process_latent_out(x)
            x = x.cpu()
            result.append(x)

        latent_image, noise = a1111_compat.KSamplerAdvanced_inspire.sample(model, True, seed, adv_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, (adv_steps-steps),
                                                                           adv_steps, noise_mode, False, callback=progress_callback, scheduler_func_opt=scheduler_func_opt)

        if not omit_final_latent:
            result.append(latent_image['samples'].cpu())

        if len(result) > 0:
            result = torch.cat(result)
            result = {'samples': result}
        else:
            result = latent_image

        return latent_image, result


class KSamplerAdvanced_progress(a1111_compat.KSamplerAdvanced_inspire):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
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
                    "omit_final_latent": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                    },
                "optional": {
                    "prev_progress_latent_opt": ("LATENT",),
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    FUNCTION = "doit"

    CATEGORY = "InspirePack/analysis"

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "progress_latent")

    def doit(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
             start_at_step, end_at_step, noise_mode, return_with_leftover_noise, interval, omit_start_latent, omit_final_latent,
             prev_progress_latent_opt=None, scheduler_func_opt=None):

        if omit_start_latent:
            result = []
        else:
            result = [latent_image['samples']]

        def progress_callback(step, x0, x, total_steps):
            if (total_steps-1) != step and step % interval != 0:
                return

            x = model.model.process_latent_out(x)
            x = x.cpu()
            result.append(x)

        latent_image, noise = a1111_compat.KSamplerAdvanced_inspire.sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step,
                                                                           noise_mode, return_with_leftover_noise, callback=progress_callback, scheduler_func_opt=scheduler_func_opt)

        if not omit_final_latent:
            result.append(latent_image['samples'].cpu())

        if len(result) > 0:
            result = torch.cat(result)
            result = {'samples': result}
        else:
            result = latent_image

        if prev_progress_latent_opt is not None:
            result['samples'] = torch.cat((prev_progress_latent_opt['samples'], result['samples']), dim=0)

        return latent_image, result


def exponential_interpolation(from_cfg, to_cfg, i, steps):
    if i == steps-1:
        return to_cfg

    if from_cfg == to_cfg:
        return from_cfg

    if from_cfg == 0:
        return to_cfg * (1 - math.exp(-5 * i / steps)) / (1 - math.exp(-5))
    elif to_cfg == 0:
        return from_cfg * (math.exp(-5 * i / steps) - math.exp(-5)) / (1 - math.exp(-5))
    else:
        log_from = math.log(from_cfg)
        log_to = math.log(to_cfg)
        log_value = log_from + (log_to - log_from) * i / steps
        return math.exp(log_value)


def logarithmic_interpolation(from_cfg, to_cfg, i, steps):
    if i == 0:
        return from_cfg

    if i == steps-1:
        return to_cfg

    log_i = math.log(i + 1)
    log_steps = math.log(steps + 1)

    t = log_i / log_steps

    return from_cfg + (to_cfg - from_cfg) * t


def cosine_interpolation(from_cfg, to_cfg, i, steps):
    if (i == 0) or (i == steps-1):
        return from_cfg

    t = (1.0 + math.cos(math.pi*2*(i/steps))) / 2

    return from_cfg + (to_cfg - from_cfg) * t


class Guider_scheduled(CFGGuider):
    def __init__(self, model_patcher, sigmas, from_cfg, to_cfg, schedule):
        super().__init__(model_patcher)
        self.default_cfg = self.cfg
        self.sigmas = sigmas
        self.cfg_sigmas = None
        self.cfg_sigmas_i = None
        self.from_cfg = from_cfg
        self.to_cfg = to_cfg
        self.schedule = schedule
        self.last_i = 0
        self.renew_cfg_sigmas()

    def set_cfg(self, cfg):
        self.default_cfg = cfg
        self.renew_cfg_sigmas()

    def renew_cfg_sigmas(self):
        self.cfg_sigmas = {}
        self.cfg_sigmas_i = {}
        i = 0
        steps = len(self.sigmas) - 1
        for x in self.sigmas:
            k = float(x)
            delta = self.to_cfg - self.from_cfg
            if self.schedule == 'exp':
                self.cfg_sigmas[k] = exponential_interpolation(self.from_cfg, self.to_cfg, i, steps), i
            elif self.schedule == 'log':
                self.cfg_sigmas[k] = logarithmic_interpolation(self.from_cfg, self.to_cfg, i, steps), i
            elif self.schedule == 'cos':
                self.cfg_sigmas[k] = cosine_interpolation(self.from_cfg, self.to_cfg, i, steps), i
            else:
                self.cfg_sigmas[k] = self.from_cfg + delta * i / steps, i

            self.cfg_sigmas_i[i] = self.cfg_sigmas[k]
            i += 1

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        k = float(timestep[0])

        v = self.cfg_sigmas.get(k)
        if v is None:
            # fallback
            v = self.cfg_sigmas_i[self.last_i+1]
            self.cfg_sigmas[k] = v

        self.last_i = v[1]
        self.cfg = v[0]

        return super().predict_noise(x, timestep, model_options, seed)


class Guider_PerpNeg_scheduled(Guider_PerpNeg):
    def __init__(self, model_patcher, sigmas, from_cfg, to_cfg, schedule, neg_scale):
        super().__init__(model_patcher)
        self.default_cfg = self.cfg
        self.sigmas = sigmas
        self.cfg_sigmas = None
        self.cfg_sigmas_i = None
        self.from_cfg = from_cfg
        self.to_cfg = to_cfg
        self.schedule = schedule
        self.neg_scale = neg_scale
        self.last_i = 0
        self.renew_cfg_sigmas()

    def set_cfg(self, cfg):
        self.default_cfg = cfg
        self.renew_cfg_sigmas()

    def renew_cfg_sigmas(self):
        self.cfg_sigmas = {}
        self.cfg_sigmas_i = {}
        i = 0
        steps = len(self.sigmas) - 1
        for x in self.sigmas:
            k = float(x)
            delta = self.to_cfg - self.from_cfg
            if self.schedule == 'exp':
                self.cfg_sigmas[k] = exponential_interpolation(self.from_cfg, self.to_cfg, i, steps), i
            elif self.schedule == 'log':
                self.cfg_sigmas[k] = logarithmic_interpolation(self.from_cfg, self.to_cfg, i, steps), i
            elif self.schedule == 'cos':
                self.cfg_sigmas[k] = cosine_interpolation(self.from_cfg, self.to_cfg, i, steps), i
            else:
                self.cfg_sigmas[k] = self.from_cfg + delta * i / steps, i

            self.cfg_sigmas_i[i] = self.cfg_sigmas[k]
            i += 1

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        k = float(timestep[0])

        v = self.cfg_sigmas.get(k)
        if v is None:
            # fallback
            v = self.cfg_sigmas_i[self.last_i+1]
            self.cfg_sigmas[k] = v

        self.last_i = v[1]
        self.cfg = v[0]

        return super().predict_noise(x, timestep, model_options, seed)


class ScheduledCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sigmas": ("SIGMAS", ),
                    "from_cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "to_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "schedule": (["linear", "log", "exp", "cos"], {'default': 'log'})
                    }
                }

    RETURN_TYPES = ("GUIDER", "SIGMAS")

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, sigmas, from_cfg, to_cfg, schedule):
        guider = Guider_scheduled(model, sigmas, from_cfg, to_cfg, schedule)
        guider.set_conds(positive, negative)
        return guider, sigmas


class ScheduledPerpNegCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "empty_conditioning": ("CONDITIONING", ),
                    "neg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                    "sigmas": ("SIGMAS", ),
                    "from_cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "to_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "schedule": (["linear", "log", "exp", "cos"], {'default': 'log'})
                    }
                }

    RETURN_TYPES = ("GUIDER", "SIGMAS")

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, empty_conditioning, neg_scale, sigmas, from_cfg, to_cfg, schedule):
        guider = Guider_PerpNeg_scheduled(model, sigmas, from_cfg, to_cfg, schedule, neg_scale)
        guider.set_conds(positive, negative, empty_conditioning)
        return guider, sigmas


NODE_CLASS_MAPPINGS = {
    "KSamplerProgress //Inspire": KSampler_progress,
    "KSamplerAdvancedProgress //Inspire": KSamplerAdvanced_progress,
    "ScheduledCFGGuider //Inspire": ScheduledCFGGuider,
    "ScheduledPerpNegCFGGuider //Inspire": ScheduledPerpNegCFGGuider
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerProgress //Inspire": "KSampler Progress (Inspire)",
    "KSamplerAdvancedProgress //Inspire": "KSampler Advanced Progress (Inspire)",
    "ScheduledCFGGuider //Inspire": "Scheduled CFGGuider (Inspire)",
    "ScheduledPerpNegCFGGuider //Inspire": "Scheduled PerpNeg CFGGuider (Inspire)"
}
