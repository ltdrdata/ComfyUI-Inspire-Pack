import itertools
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


def apply_variation_noise(latent_image, noise_device, variation_seed, variation_strength, mask=None):
    latent_size = latent_image.size()
    latent_size_1batch = [1, latent_size[1], latent_size[2], latent_size[3]]

    if noise_device == "cpu":
        variation_generator = torch.manual_seed(variation_seed)
    else:
        torch.cuda.manual_seed(variation_seed)
        variation_generator = None

    variation_latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                   generator=variation_generator, device=noise_device)

    variation_noise = variation_latent.expand(latent_image.size()[0], -1, -1, -1)

    if variation_strength == 0:
        return latent_image
    elif mask is None:
        result = (1 - variation_strength) * latent_image + variation_strength * variation_noise
    else:
        # this seems precision is not enough when variation_strength is 0.0
        result = (mask == 1).float() * ((1 - variation_strength) * latent_image + variation_strength * variation_noise * mask) + (mask == 0).float() * latent_image

    return result


def prepare_noise(latent_image, seed, noise_inds=None, noise_device="cpu", incremental_seed_mode="comfy", variation_seed=None, variation_strength=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """

    latent_size = latent_image.size()
    latent_size_1batch = [1, latent_size[1], latent_size[2], latent_size[3]]

    if variation_strength is not None and variation_strength > 0 or incremental_seed_mode.startswith("variation str inc"):
        if noise_device == "cpu":
            variation_generator = torch.manual_seed(variation_seed)
        else:
            torch.cuda.manual_seed(variation_seed)
            variation_generator = None

        variation_latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                       generator=variation_generator, device=noise_device)
    else:
        variation_latent = None

    def apply_variation(input_latent, strength_up=None):
        if variation_latent is None:
            return input_latent
        else:
            strength = variation_strength

            if strength_up is not None:
                strength += strength_up

            variation_noise = variation_latent.expand(input_latent.size()[0], -1, -1, -1)
            result = (1 - strength) * input_latent + strength * variation_noise
            return result

    # method: incremental seed batch noise
    if noise_inds is None and incremental_seed_mode == "incremental":
        batch_cnt = latent_size[0]

        latents = None
        for i in range(batch_cnt):
            if noise_device == "cpu":
                generator = torch.manual_seed(seed+i)
            else:
                torch.cuda.manual_seed(seed+i)
                generator = None

            latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                 generator=generator, device=noise_device)

            latent = apply_variation(latent)

            if latents is None:
                latents = latent
            else:
                latents = torch.cat((latents, latent), dim=0)

        return latents

    # method: incremental variation batch noise
    elif noise_inds is None and incremental_seed_mode.startswith("variation str inc"):
        batch_cnt = latent_size[0]

        latents = None
        for i in range(batch_cnt):
            if noise_device == "cpu":
                generator = torch.manual_seed(seed)
            else:
                torch.cuda.manual_seed(seed)
                generator = None

            latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout,
                                 generator=generator, device=noise_device)

            step = float(incremental_seed_mode[18:])
            latent = apply_variation(latent, step*i)

            if latents is None:
                latents = latent
            else:
                latents = torch.cat((latents, latent), dim=0)

        return latents

    # method: comfy batch noise
    if noise_device == "cpu":
        generator = torch.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
        generator = None

    if noise_inds is None:
        latents = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                              generator=generator, device=noise_device)
        latents = apply_variation(latents)
        return latents

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout,
                            generator=generator, device=noise_device)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def empty_pil_tensor(w=64, h=64):
    image = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, w-1, h-1), fill=(0, 0, 0))
    return pil2tensor(image)


def try_install_custom_node(custom_node_url, msg):
    try:
        import cm_global
        cm_global.try_call(api='cm.try-install-custom-node',
                           sender="Inspire Pack", custom_node_url=custom_node_url, msg=msg)
    except Exception as e:
        print(msg)
        print(f"[Inspire Pack] ComfyUI-Manager is outdated. The custom node installation feature is not available.")


def empty_latent():
    return torch.zeros([1, 4, 8, 8])

# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


# author: Trung0246 --->
class TautologyStr(str):
    def __ne__(self, other):
        return False


class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item


class TaggedCache:
    def __init__(self, tag_settings: Optional[dict]=None):
        self._tag_settings = tag_settings or {}  # tag cache size
        self._data = {}

    def __getitem__(self, key):
        for tag_data in self._data.values():
            if key in tag_data:
                return tag_data[key]
        raise KeyError(f'Key `{key}` does not exist')

    def __setitem__(self, key, value: tuple):
        # value: (tag: str, (islist: bool, data: *))

        # if key already exists, pop old value
        for tag_data in self._data.values():
            if key in tag_data:
                tag_data.pop(key, None)
                break

        tag = value[0]
        if tag not in self._data:

            try:
                from cachetools import LRUCache

                default_size = 20
                if 'ckpt' in tag:
                    default_size = 5
                elif tag in ['latent', 'image']:
                    default_size = 100

                self._data[tag] = LRUCache(maxsize=self._tag_settings.get(tag, default_size))

            except (ImportError, ModuleNotFoundError):
                # TODO: implement a simple lru dict
                self._data[tag] = {}
        self._data[tag][key] = value

    def __delitem__(self, key):
        for tag_data in self._data.values():
            if key in tag_data:
                del tag_data[key]
                return
        raise KeyError(f'Key `{key}` does not exist')

    def __contains__(self, key):
        return any(key in tag_data for tag_data in self._data.values())

    def items(self):
        yield from itertools.chain(*map(lambda x :x.items(), self._data.values()))

    def get(self, key, default=None):
        """D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."""
        for tag_data in self._data.values():
            if key in tag_data:
                return tag_data[key]
        return default

    def clear(self):
        # clear all cache
        self._data = {}
