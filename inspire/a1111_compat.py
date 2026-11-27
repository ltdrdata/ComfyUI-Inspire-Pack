import comfy
import torch
from .libs import utils
from einops import rearrange
import random
import math
from .libs import common
import logging


supported_noise_modes = ["GPU(=A1111)", "CPU", "GPU+internal_seed", "CPU+internal_seed"]


class Inspire_RandomNoise:
    def __init__(self, seed, mode, incremental_seed_mode, variation_seed, variation_strength, variation_method="linear", internal_seed=None):
        device = comfy.model_management.get_torch_device()
        # HOTFIX: https://github.com/comfyanonymous/ComfyUI/commit/916d1e14a93ef331adef7c0deff2fdcf443b05cf#commitcomment-151914788
        # seed value should be different with generated noise

        self.seed = internal_seed
        self.noise_seed = seed
        self.noise_device = "cpu" if mode == "CPU" else device
        self.incremental_seed_mode = incremental_seed_mode
        self.variation_seed = variation_seed
        self.variation_strength = variation_strength
        self.variation_method = variation_method

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        noise = utils.prepare_noise(latent_image, self.noise_seed, batch_inds, self.noise_device, self.incremental_seed_mode,
                                    variation_seed=self.variation_seed, variation_strength=self.variation_strength, variation_method=self.variation_method)
        return noise.cpu()


class RandomNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "This is the seed for the initial noise applied to the latent."}),
                    "noise_mode": (["GPU(=A1111)", "CPU"],),
                    "batch_seed_mode": (["incremental", "comfy", "variation str inc:0.01", "variation str inc:0.05"],),
                    "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                "optional":
                    {
                        "variation_method": (["linear", "slerp"],),
                        "internal_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "This is the seed used for generating noise in intermediate steps when using ancestral and SDE-based samplers.\nNOTE: If `noise_mode` is in GPU mode and `internal_seed` is the same as `seed`, the generated image may be distorted."}),
                    }
                }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "InspirePack/a1111_compat"

    def get_noise(self, noise_seed, noise_mode, batch_seed_mode, variation_seed, variation_strength, variation_method="linear", internal_seed=None):
        if internal_seed is None:
            internal_seed = noise_seed

        return (Inspire_RandomNoise(noise_seed, noise_mode, batch_seed_mode, variation_seed, variation_strength, variation_method=variation_method, internal_seed=internal_seed),)


def inspire_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                     noise_mode="CPU", disable_noise=False, start_step=None, last_step=None, force_full_denoise=False,
                     incremental_seed_mode="comfy", variation_seed=None, variation_strength=None, noise=None, callback=None, variation_method="linear",
                     scheduler_func=None, internal_seed=None):
    device = comfy.model_management.get_torch_device()
    noise_device = "cpu" if 'cpu' in noise_mode.lower() else device
    latent_image = latent["samples"]
    if hasattr(comfy.sample, 'fix_empty_latent_channels'):
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    latent = latent.copy()

    if noise is not None and latent_image.shape[1] != noise.shape[1]:
        logging.info("[Inspire Pack] inspire_ksampler: The type of latent input for noise generation does not match the model's latent type. When using the SD3 model, you must use the SD3 Empty Latent.")
        raise Exception("The type of latent input for noise generation does not match the model's latent type. When using the SD3 model, you must use the SD3 Empty Latent.")

    if noise is None:
        if disable_noise:
            torch.manual_seed(seed)
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=noise_device)
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = utils.prepare_noise(latent_image, seed, batch_inds, noise_device, incremental_seed_mode,
                                        variation_seed=variation_seed, variation_strength=variation_strength, variation_method=variation_method)

    if start_step is None:
        if denoise == 1.0:
            start_step = 0
        else:
            advanced_steps = math.floor(steps / denoise)
            start_step = advanced_steps - steps
            steps = advanced_steps

    if internal_seed is None:
        internal_seed = seed

    if 'internal_seed' in noise_mode:
        seed = internal_seed

    try:
        samples = common.impact_sampling(
            model=model, add_noise=not disable_noise, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative,
            latent_image=latent, start_at_step=start_step, end_at_step=last_step, return_with_leftover_noise=not force_full_denoise, noise=noise, callback=callback,
            scheduler_func=scheduler_func)
    except Exception as e:
        if "unexpected keyword argument 'scheduler_func'" in str(e):
            logging.info("[Inspire Pack] Impact Pack is outdated. (Cannot use GITS scheduler.)")

            samples = common.impact_sampling(
                model=model, add_noise=not disable_noise, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive, negative=negative,
                latent_image=latent, start_at_step=start_step, end_at_step=last_step, return_with_leftover_noise=not force_full_denoise, noise=noise, callback=callback)
        else:
            raise e

    return samples, noise


class KSampler_inspire:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "This is the seed for the initial noise applied to the latent."}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (common.get_schedulers(), ),
                     "positive": ("CONDITIONING", ),
                     "negative": ("CONDITIONING", ),
                     "latent_image": ("LATENT", ),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "noise_mode": (supported_noise_modes,),
                     "batch_seed_mode": (["incremental", "comfy", "variation str inc:0.01", "variation str inc:0.05"],),
                     "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "optional":
                    {
                        "variation_method": (["linear", "slerp"],),
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                        "internal_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "This is the seed used for generating noise in intermediate steps when using ancestral and SDE-based samplers.\nNOTE: If `noise_mode` is in GPU mode and `internal_seed` is the same as `seed`, the generated image may be distorted."}),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/a1111_compat"

    @staticmethod
    def doit(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, noise_mode,
             batch_seed_mode="comfy", variation_seed=None, variation_strength=None, variation_method="linear", scheduler_func_opt=None,
             internal_seed=None):
        return (inspire_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, noise_mode,
                                 incremental_seed_mode=batch_seed_mode, variation_seed=variation_seed, variation_strength=variation_strength, variation_method=variation_method,
                                 scheduler_func=scheduler_func_opt, internal_seed=internal_seed)[0], )


class KSamplerAdvanced_inspire:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "This is the seed for the initial noise applied to the latent."}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (common.get_schedulers(), ),
                     "positive": ("CONDITIONING", ),
                     "negative": ("CONDITIONING", ),
                     "latent_image": ("LATENT", ),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "noise_mode": (supported_noise_modes,),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                     "batch_seed_mode": (["incremental", "comfy", "variation str inc:0.01", "variation str inc:0.05"],),
                     "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "optional":
                    {
                        "variation_method": (["linear", "slerp"],),
                        "noise_opt": ("NOISE_IMAGE",),
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                        "internal_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "This is the seed used for generating noise in intermediate steps when using ancestral and SDE-based samplers.\nNOTE: If `noise_mode` is in GPU mode and `internal_seed` is the same as `seed`, the generated image may be distorted."}),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/a1111_compat"

    @staticmethod
    def sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, noise_mode, return_with_leftover_noise,
               denoise=1.0, batch_seed_mode="comfy", variation_seed=None, variation_strength=None, noise_opt=None, callback=None, variation_method="linear", scheduler_func_opt=None, internal_seed=None):
        force_full_denoise = True

        if return_with_leftover_noise:
            force_full_denoise = False

        disable_noise = False

        if not add_noise:
            disable_noise = True

        return inspire_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step,
                                force_full_denoise=force_full_denoise, noise_mode=noise_mode, incremental_seed_mode=batch_seed_mode,
                                variation_seed=variation_seed, variation_strength=variation_strength, noise=noise_opt, callback=callback, variation_method=variation_method,
                                scheduler_func=scheduler_func_opt, internal_seed=internal_seed)

    def doit(self, *args, **kwargs):
        return (self.sample(*args, **kwargs)[0],)


class KSampler_inspire_pipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "This is the seed for the initial noise applied to the latent."}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (common.get_schedulers(), ),
                     "latent_image": ("LATENT", ),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "noise_mode": (supported_noise_modes,),
                     "batch_seed_mode": (["incremental", "comfy", "variation str inc:0.01", "variation str inc:0.05"],),
                     "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "optional":
                    {
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                        "internal_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "This is the seed used for generating noise in intermediate steps when using ancestral and SDE-based samplers.\nNOTE: If `noise_mode` is in GPU mode and `internal_seed` is the same as `seed`, the generated image may be distorted."}),
                    }
                }

    RETURN_TYPES = ("LATENT", "VAE")
    FUNCTION = "sample"

    CATEGORY = "InspirePack/a1111_compat"

    def sample(self, basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise, noise_mode, batch_seed_mode="comfy",
               variation_seed=None, variation_strength=None, scheduler_func_opt=None, internal_seed=None):
        model, clip, vae, positive, negative = basic_pipe
        latent = inspire_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, noise_mode, incremental_seed_mode=batch_seed_mode,
                                  variation_seed=variation_seed, variation_strength=variation_strength, scheduler_func=scheduler_func_opt, internal_seed=internal_seed)[0]
        return latent, vae


class KSamplerAdvanced_inspire_pipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "This is the seed for the initial noise applied to the latent."}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (common.get_schedulers(), ),
                     "latent_image": ("LATENT", ),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "noise_mode": (supported_noise_modes,),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                     "batch_seed_mode": (["incremental", "comfy", "variation str inc:0.01", "variation str inc:0.05"],),
                     "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "optional":
                    {
                        "noise_opt": ("NOISE_IMAGE",),
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                        "internal_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "This is the seed used for generating noise in intermediate steps when using ancestral and SDE-based samplers.\nNOTE: If `noise_mode` is in GPU mode and `internal_seed` is the same as `seed`, the generated image may be distorted."}),
                    }
                }

    RETURN_TYPES = ("LATENT", "VAE", )
    FUNCTION = "sample"

    CATEGORY = "InspirePack/a1111_compat"

    def sample(self, basic_pipe, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, latent_image, start_at_step, end_at_step, noise_mode, return_with_leftover_noise,
               denoise=1.0, batch_seed_mode="comfy", variation_seed=None, variation_strength=None, noise_opt=None, scheduler_func_opt=None, internal_seed=None):
        model, clip, vae, positive, negative = basic_pipe
        latent = KSamplerAdvanced_inspire().sample(model=model, add_noise=add_noise, noise_seed=noise_seed,
                                                   steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                                                   positive=positive, negative=negative, latent_image=latent_image,
                                                   start_at_step=start_at_step, end_at_step=end_at_step,
                                                   noise_mode=noise_mode, return_with_leftover_noise=return_with_leftover_noise,
                                                   denoise=denoise, batch_seed_mode=batch_seed_mode, variation_seed=variation_seed,
                                                   variation_strength=variation_strength, noise_opt=noise_opt, scheduler_func_opt=scheduler_func_opt,
                                                   internal_seed=internal_seed)[0]
        return latent, vae


# Modified version of ComfyUI main code
# https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_hypertile.py
def get_closest_divisors(hw: int, aspect_ratio: float) -> tuple[int, int]:
    pairs = [(i, hw // i) for i in range(int(math.sqrt(hw)), 1, -1) if hw % i == 0]
    pair = min(((i, hw // i) for i in range(2, hw + 1) if hw % i == 0),
               key=lambda x: abs(x[1] / x[0] - aspect_ratio))
    pairs.append(pair)
    res = min(pairs, key=lambda x: max(x) / min(x))
    return res


def calc_optimal_hw(hw: int, aspect_ratio: float) -> tuple[int, int]:
    hcand = round(math.sqrt(hw * aspect_ratio))
    wcand = hw // hcand

    if hcand * wcand != hw:
        wcand = round(math.sqrt(hw / aspect_ratio))
        hcand = hw // wcand

        if hcand * wcand != hw:
            return get_closest_divisors(hw, aspect_ratio)

    return hcand, wcand


def random_divisor(value: int, min_value: int, /, max_options: int = 1, rand_obj=random.Random()) -> int:
    # print(f"value={value}, min_value={min_value}, max_options={max_options}")
    min_value = min(min_value, value)

    # All big divisors of value (inclusive)
    divisors = [i for i in range(min_value, value + 1) if value % i == 0]

    ns = [value // i for i in divisors[:max_options]]  # has at least 1 element

    if len(ns) - 1 > 0:
        idx = rand_obj.randint(0, len(ns) - 1)
    else:
        idx = 0
    # print(f"ns={ns}, idx={idx}")

    return ns[idx]

# def get_divisors(value: int, min_value: int, /, max_options: int = 1) -> list[int]:
#     """
#     Returns divisors of value that
#         x * min_value <= value
#     in big -> small order, amount of divisors is limited by max_options
#     """
#     max_options = max(1, max_options) # at least 1 option should be returned
#     min_value = min(min_value, value)
#     divisors = [i for i in range(min_value, value + 1) if value % i == 0] # divisors in small -> big order
#     ns = [value // i for i in divisors[:max_options]]  # has at least 1 element # big -> small order
#     return ns


# def random_divisor(value: int, min_value: int, /, max_options: int = 1, rand_obj=None) -> int:
#     """
#     Returns a random divisor of value that
#         x * min_value <= value
#     if max_options is 1, the behavior is deterministic
#     """
#     print(f"value={value}, min_value={min_value}, max_options={max_options}")
#     ns = get_divisors(value, min_value, max_options=max_options) # get cached divisors
#     idx = rand_obj.randint(0, len(ns) - 1)
#     print(f"ns={ns}, idx={idx}")
#
#     return ns[idx]


class HyperTileInspire:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "tile_size": ("INT", {"default": 256, "min": 1, "max": 2048}),
                             "swap_size": ("INT", {"default": 2, "min": 1, "max": 128}),
                             "max_depth": ("INT", {"default": 0, "min": 0, "max": 10}),
                             "scale_depth": ("BOOLEAN", {"default": False}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "InspirePack/__for_testing"

    def patch(self, model, tile_size, swap_size, max_depth, scale_depth, seed):
        latent_tile_size = max(32, tile_size) // 8
        temp = None

        rand_obj = random.Random()
        rand_obj.seed(seed)

        def hypertile_in(q, k, v, extra_options):
            nonlocal temp
            model_chans = q.shape[-2]
            orig_shape = extra_options['original_shape']
            apply_to = []
            for i in range(max_depth + 1):
                apply_to.append((orig_shape[-2] / (2 ** i)) * (orig_shape[-1] / (2 ** i)))

            if model_chans in apply_to:
                shape = extra_options["original_shape"]
                aspect_ratio = shape[-1] / shape[-2]

                hw = q.size(1)
                # h, w = calc_optimal_hw(hw, aspect_ratio)
                h, w = round(math.sqrt(hw * aspect_ratio)), round(math.sqrt(hw / aspect_ratio))

                factor = (2 ** apply_to.index(model_chans)) if scale_depth else 1
                nh = random_divisor(h, latent_tile_size * factor, swap_size, rand_obj)
                nw = random_divisor(w, latent_tile_size * factor, swap_size, rand_obj)

                logging.debug(f"factor: {factor} <--- params.depth: {apply_to.index(model_chans)} / scale_depth: {scale_depth} / latent_tile_size={latent_tile_size}")
                # print(f"h: {h}, w:{w} --> nh: {nh}, nw: {nw}")

                if nh * nw > 1:
                    q = rearrange(q, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)
                    temp = (nh, nw, h, w)
                # else:
                #     temp = None

                logging.debug(f"q={q} / k={k} / v={v}")
                return q, k, v

            return q, k, v

        def hypertile_out(out, extra_options):
            nonlocal temp
            if temp is not None:
                nh, nw, h, w = temp
                temp = None
                out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)
            return out

        m = model.clone()
        m.set_model_attn1_patch(hypertile_in)
        m.set_model_attn1_output_patch(hypertile_out)
        return (m, )


NODE_CLASS_MAPPINGS = {
    "KSampler //Inspire": KSampler_inspire,
    "KSamplerAdvanced //Inspire": KSamplerAdvanced_inspire,
    "KSamplerPipe //Inspire": KSampler_inspire_pipe,
    "KSamplerAdvancedPipe //Inspire": KSamplerAdvanced_inspire_pipe,
    "RandomNoise //Inspire": RandomNoise,
    "HyperTile //Inspire": HyperTileInspire
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSampler //Inspire": "KSampler (inspire)",
    "KSamplerAdvanced //Inspire": "KSamplerAdvanced (inspire)",
    "KSamplerPipe //Inspire": "KSampler [pipe] (inspire)",
    "KSamplerAdvancedPipe //Inspire": "KSamplerAdvanced [pipe] (inspire)",
    "RandomNoise //Inspire": "RandomNoise (inspire)",
    "HyperTile //Inspire": "HyperTile (Inspire)"
}
