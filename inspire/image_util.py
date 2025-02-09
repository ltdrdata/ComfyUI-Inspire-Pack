import os

import torch
from PIL import ImageOps
import comfy
import folder_paths
import base64
from io import BytesIO
from .libs.utils import *
import json


class LoadImagesFromDirBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                # XXX TODO add a switch for the size normalization
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("images", "masks", "count", "image_paths_json")
    FUNCTION = "load_images"

    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.'")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        image_paths = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        has_non_empty_mask = False

        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                has_non_empty_mask = True
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            image_paths.append(image_path)
            image_count += 1

        if len(images) == 1:
            return (images[0], masks[0], 1, json.dumps(image_paths), )

        elif len(images) > 1:
            image1 = images[0]
            mask1 = None

            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks:
                if has_non_empty_mask:
                    if image1.shape[1:3] != mask2.shape:
                        mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(image1.shape[1], image1.shape[2]), mode='bilinear', align_corners=False)
                        mask2 = mask2.squeeze(0)
                    else:
                        mask2 = mask2.unsqueeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)

                if mask1 is None:
                    mask1 = mask2
                else:
                    mask1 = torch.cat((mask1, mask2), dim=0)

            return (image1, mask1, len(images), json.dumps(image_paths), )


class LoadImagesFromDirList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "FILE PATH")
    OUTPUT_IS_LIST = (True, True, True)

    FUNCTION = "load_images"

    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        file_paths = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            images.append(image)
            masks.append(mask)
            file_paths.append(str(image_path))
            image_count += 1

        return (images, masks, file_paths)


class LoadImageInspire:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {
                                "image": (sorted(files) + ["#DATA"], {"image_upload": True}),
                                "image_data": ("STRING", {"multiline": False}),
                            }
                }

    CATEGORY = "InspirePack/image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image, image_data):
        image_data = base64.b64decode(image_data.split(",")[1])
        i = Image.open(BytesIO(image_data))
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0))


class ChangeImageBatchSize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "image": ("IMAGE",),
                                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                                "mode": (["simple"],)
                            }
                }

    CATEGORY = "InspirePack/Util"

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    @staticmethod
    def resize_tensor(input_tensor, batch_size, mode):
        if mode == "simple":
            if len(input_tensor) < batch_size:
                last_frame = input_tensor[-1].unsqueeze(0).expand(batch_size - len(input_tensor), -1, -1, -1)
                output_tensor = torch.concat((input_tensor, last_frame), dim=0)
            else:
                output_tensor = input_tensor[:batch_size, :, :, :]
            return output_tensor
        else:
            print(f"[WARN] ChangeImage(Latent)BatchSize: Unknown mode `{mode}` - ignored")
            return input_tensor

    @staticmethod
    def doit(image, batch_size, mode):
        res = ChangeImageBatchSize.resize_tensor(image, batch_size, mode)
        return (res,)


class ChangeLatentBatchSize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "latent": ("LATENT",),
                                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                                "mode": (["simple"],)
                            }
                }

    CATEGORY = "InspirePack/Util"

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    @staticmethod
    def doit(latent, batch_size, mode):
        res_latent = latent.copy()
        samples = res_latent['samples']
        samples = ChangeImageBatchSize.resize_tensor(samples, batch_size, mode)
        res_latent['samples'] = samples
        return (res_latent,)


class ImageBatchSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "split_count": ("INT", {"default": 4, "min": 0, "max": 50, "step": 1}),
                    },
                }

    RETURN_TYPES = ByPassTypeTuple(("IMAGE", ))
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, images, split_count):
        cnt = min(split_count, len(images))
        res = [image.unsqueeze(0) for image in images[:cnt]]

        if split_count >= len(images):
            lack_cnt = split_count - cnt + 1  # including remained
            empty_image = empty_pil_tensor()
            for x in range(0, lack_cnt):
                res.append(empty_image)
        elif cnt < len(images):
            remained_cnt = len(images) - cnt
            remained_image = images[-remained_cnt:]
            res.append(remained_image)

        return tuple(res)


class LatentBatchSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "split_count": ("INT", {"default": 4, "min": 0, "max": 50, "step": 1}),
                    },
                }

    RETURN_TYPES = ByPassTypeTuple(("LATENT", ))
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, latent, split_count):
        samples = latent['samples']

        latent_base = latent.copy()
        del latent_base['samples']

        cnt = min(split_count, len(samples))
        res = []

        for single_samples in samples[:cnt]:
            item = latent_base.copy()
            item['samples'] = single_samples.unsqueeze(0)
            res.append(item)

        if split_count >= len(samples):
            lack_cnt = split_count - cnt + 1  # including remained
            item = latent_base.copy()
            item['samples'] = empty_latent()

            for x in range(0, lack_cnt):
                res.append(item)

        elif cnt < len(samples):
            remained_cnt = len(samples) - cnt
            remained_latent = latent_base.copy()
            remained_latent['samples'] = samples[-remained_cnt:]
            res.append(remained_latent)

        return tuple(res)


def top_k_colors(image_tensor, k, min_pixels):
    flattened_image = image_tensor.view(-1, image_tensor.size(-1))

    unique_colors, counts = torch.unique(flattened_image, dim=0, return_counts=True)

    sorted_counts, sorted_indices = torch.sort(counts, descending=True)
    sorted_colors = unique_colors[sorted_indices]

    filtered_colors = sorted_colors[sorted_counts >= min_pixels]

    return filtered_colors[:k]


def create_mask(image_tensor, color):
    mask_tensor = torch.zeros_like(image_tensor[:, :, :, 0])
    mask_tensor = torch.where(torch.all(image_tensor == color, dim=-1, keepdim=False), 1, mask_tensor)
    return mask_tensor


class ColorMapToMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "color_map": ("IMAGE",),
                    "min_pixels": ("INT", {"default": 500, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
                    "max_count": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
                    },
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, color_map, max_count, min_pixels):
        if len(color_map) > 0:
            print(f"[Inspire Pack] WARN: ColorMapToMasks - Sure, here's the translation: `color_map` can only be a single image. Only the first image will be processed. If you want to utilize the remaining images, convert the Image Batch to an Image List.")

        top_colors = top_k_colors(color_map[0], max_count, min_pixels)

        masks = None

        for color in top_colors:
            this_mask = create_mask(color_map, color)
            if masks is None:
                masks = this_mask
            else:
                masks = torch.concat((masks, this_mask), dim=0)

        if masks is None:
            masks = torch.zeros_like(color_map[0, :, :, 0])
            masks.unsqueeze(0)

        return (masks,)


class SelectNthMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "masks": ("MASK",),
                    "idx": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                    },
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"
    CATEGORY = "InspirePack/Util"

    def doit(self, masks, idx):
        return (masks[idx].unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "LoadImagesFromDir //Inspire": LoadImagesFromDirBatch,
    "LoadImageListFromDir //Inspire": LoadImagesFromDirList,
    "LoadImage //Inspire": LoadImageInspire,
    "ChangeImageBatchSize //Inspire": ChangeImageBatchSize,
    "ChangeLatentBatchSize //Inspire": ChangeLatentBatchSize,
    "ImageBatchSplitter //Inspire": ImageBatchSplitter,
    "LatentBatchSplitter //Inspire": LatentBatchSplitter,
    "ColorMapToMasks //Inspire": ColorMapToMasks,
    "SelectNthMask //Inspire": SelectNthMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagesFromDir //Inspire": "Load Image Batch From Dir (Inspire)",
    "LoadImageListFromDir //Inspire": "Load Image List From Dir (Inspire)",
    "LoadImage //Inspire": "Load Image (Inspire)",
    "ChangeImageBatchSize //Inspire": "Change Image Batch Size (Inspire)",
    "ChangeLatentBatchSize //Inspire": "Change Latent Batch Size (Inspire)",
    "ImageBatchSplitter //Inspire": "Image Batch Splitter (Inspire)",
    "LatentBatchSplitter //Inspire": "Latent Batch Splitter (Inspire)",
    "ColorMapToMasks //Inspire": "Color Map To Masks (Inspire)",
    "SelectNthMask //Inspire": "Select Nth Mask (Inspire)"
}
