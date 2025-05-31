import os

import torch
from PIL import ImageOps
try:
    import pillow_jxl      # noqa: F401
    jxl = True
except ImportError:
    jxl = False
import comfy
import folder_paths
import base64
from io import BytesIO
from .libs.utils import ByPassTypeTuple, empty_pil_tensor, empty_latent
from PIL import Image
import numpy as np
import logging
import json

try:
    from pypdf import PdfReader
    pdf_support = True
except ImportError:
    pdf_support = False

try:
    from docx import Document
    docx_support = True
except ImportError:
    docx_support = False

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
                "sort_criteria": (["filename", "newest_first", "oldest_first"], {"default": "filename"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    FUNCTION = "load_images"

    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs.items()))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False, sort_criteria="filename"):
        if not os.path.isdir(directory):
            # It's generally better to return empty/default values if the node is part of a larger workflow
            # rather than raising an error that stops the entire queue, unless the error is critical.
            logging.error(f"Directory '{directory}' cannot be found. Returning empty tensors.")
            return (empty_pil_tensor(), empty_pil_tensor(channels=1).squeeze(-1), 0)

        try:
            dir_files_short = os.listdir(directory)
        except OSError as e:
            logging.error(f"Error listing directory '{directory}': {e}. Returning empty tensors.")
            return (empty_pil_tensor(), empty_pil_tensor(channels=1).squeeze(-1), 0)

        if not dir_files_short:
            logging.warning(f"No files in directory '{directory}'. Returning empty tensors.")
            return (empty_pil_tensor(), empty_pil_tensor(channels=1).squeeze(-1), 0)

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        if jxl:
            valid_extensions.extend('.jxl')
        
        processed_files = []
        for f_short in dir_files_short:
            if any(f_short.lower().endswith(ext) for ext in valid_extensions):
                full_path = os.path.join(directory, f_short)
                if not os.path.isdir(full_path): # Ensure it's a file
                    processed_files.append(full_path)
        
        if not processed_files:
            logging.warning(f"No valid image files found in directory '{directory}' after filtering. Returning empty tensors.")
            return (empty_pil_tensor(), empty_pil_tensor(channels=1).squeeze(-1), 0)

        # Sort files based on sort_criteria
        if sort_criteria == "newest_first":
            processed_files.sort(key=os.path.getmtime, reverse=True)
        elif sort_criteria == "oldest_first":
            processed_files.sort(key=os.path.getmtime)
        else:  # 'filename' or default
            processed_files.sort() # Sorts by full path, which includes filename

        # Apply start_index
        if start_index == -1:
            # Handle -1 to mean the last item if the list is not empty
            if len(processed_files) > 0:
                files_to_process = processed_files[start_index:] 
            else:
                files_to_process = []
        elif start_index < len(processed_files):
            files_to_process = processed_files[start_index:]
        else: # start_index is out of bounds
            files_to_process = []

        images = []
        masks = []

        # Cap the number of images to load
        if image_load_cap > 0:
            files_to_process = files_to_process[:image_load_cap]

        if not files_to_process:
            logging.warning(f"No images to load from '{directory}' after slicing/capping (start_index: {start_index}, cap: {image_load_cap}, sort: {sort_criteria}). Returning empty tensors.")
            return (empty_pil_tensor(), empty_pil_tensor(channels=1).squeeze(-1), 0)

        for image_path in files_to_process:
            try:
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                image_rgb = i.convert("RGB")
                image_np = np.array(image_rgb).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,] # (1, H, W, C)

                if 'A' in i.getbands():
                    mask_alpha = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = 1. - torch.from_numpy(mask_alpha) # (H, W)
                else:
                    # Create a zero mask with the same H, W as the current image
                    mask_tensor = torch.zeros((image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32, device="cpu")
                
                images.append(image_tensor)
                masks.append(mask_tensor)
            except Exception as e:
                logging.warning(f"Could not load image {image_path}: {e}")
                continue

        if not images: # If all attempts to load images failed or list was empty
            logging.warning(f"No images were successfully loaded from directory '{directory}' with current settings. Returning empty tensors.")
            return (empty_pil_tensor(), empty_pil_tensor(channels=1).squeeze(-1), 0)

        if len(images) == 1:
            # Output mask as (1, H, W) for consistency with batched output
            return (images[0], masks[0].unsqueeze(0), 1)
        elif len(images) > 1:
            ref_H, ref_W = images[0].shape[1], images[0].shape[2] # H, W from first image

            batched_images_list = [images[0]]
            for img_tensor in images[1:]: # img_tensor is (1, H_orig, W_orig, C)
                if img_tensor.shape[1] != ref_H or img_tensor.shape[2] != ref_W:
                    img_tensor_permuted = img_tensor.permute(0, 3, 1, 2) # (1, C, H_orig, W_orig)
                    img_resized_permuted = comfy.utils.common_upscale(img_tensor_permuted, ref_W, ref_H, "bilinear", "center")
                    img_resized = img_resized_permuted.permute(0, 2, 3, 1) # (1, H_ref, W_ref, C)
                    batched_images_list.append(img_resized)
                else:
                    batched_images_list.append(img_tensor)
            
            batched_images_tensor = torch.cat(batched_images_list, dim=0) # (B, H_ref, W_ref, C)

            processed_masks = []
            for m_single in masks: # m_single is a 2D tensor (H_orig, W_orig)
                if m_single.shape[0] != ref_H or m_single.shape[1] != ref_W:
                    m_reshaped = m_single.reshape((1, 1, m_single.shape[0], m_single.shape[1])) # for interpolate
                    m_resized = torch.nn.functional.interpolate(m_reshaped, size=(ref_H, ref_W), mode="bilinear", align_corners=False)
                    processed_masks.append(m_resized.squeeze(0).squeeze(0)) # Back to (H_ref, W_ref)
                else:
                    processed_masks.append(m_single)
            
            batched_masks_tensor = torch.stack(processed_masks, dim=0) # (B, H_ref, W_ref)
            
            return (batched_images_tensor, batched_masks_tensor, len(images))
        
        # Fallback for any unexpected scenario, though prior checks should prevent reaching here with empty lists
        return (empty_pil_tensor(), empty_pil_tensor(channels=1).squeeze(-1), 0)


class LoadTextBatchFromDir:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "load_cap": ("INT", {"default": 0, "min": 0, "step": 1}), # Renamed from image_load_cap
                "start_index": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_criteria": (["filename", "newest_first", "oldest_first"], {"default": "filename"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("TEXT_BATCH", "COUNT")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "load_texts"

    CATEGORY = "text" # Or "InspirePack/Text"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs.items()))

    def load_texts(self, directory: str, load_cap: int = 0, start_index: int = 0, load_always=False, sort_criteria="filename"):
        if not os.path.isdir(directory):
            logging.error(f"Directory '{directory}' cannot be found. Returning empty list.")
            return ([], 0)

        try:
            dir_files_short = os.listdir(directory)
        except OSError as e:
            logging.error(f"Error listing directory '{directory}': {e}. Returning empty list.")
            return ([], 0)

        if not dir_files_short:
            logging.warning(f"No files in directory '{directory}'. Returning empty list.")
            return ([], 0)

        valid_extensions = ['.txt', '.json']
        if pdf_support:
            valid_extensions.append('.pdf')
        else:
            logging.info("[LoadTextBatchFromDir] PyPDF2 not installed. PDF support disabled. To enable, run: pip install pypdf")
        if docx_support:
            valid_extensions.append('.docx') # Handling .docx for .doc request
        else:
            logging.info("[LoadTextBatchFromDir] python-docx not installed. DOCX support disabled. To enable, run: pip install python-docx")
        
        processed_files = []
        for f_short in dir_files_short:
            if any(f_short.lower().endswith(ext) for ext in valid_extensions):
                full_path = os.path.join(directory, f_short)
                if not os.path.isdir(full_path):
                    processed_files.append(full_path)
        
        if not processed_files:
            logging.warning(f"No valid text files found in directory '{directory}' after filtering. Returning empty list.")
            return ([], 0)

        if sort_criteria == "newest_first":
            processed_files.sort(key=os.path.getmtime, reverse=True)
        elif sort_criteria == "oldest_first":
            processed_files.sort(key=os.path.getmtime)
        else:
            processed_files.sort()

        if start_index == -1:
            files_to_process = processed_files[start_index:] if len(processed_files) > 0 else []
        elif start_index < len(processed_files):
            files_to_process = processed_files[start_index:]
        else:
            files_to_process = []

        if load_cap > 0:
            files_to_process = files_to_process[:load_cap]

        if not files_to_process:
            logging.warning(f"No text files to load from '{directory}' after slicing/capping. Returning empty list.")
            return ([], 0)

        texts = []
        for file_path in files_to_process:
            content = ""
            try:
                if file_path.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif file_path.lower().endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        content = json.dumps(data, indent=2)
                elif pdf_support and file_path.lower().endswith('.pdf'):
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                elif docx_support and file_path.lower().endswith('.docx'):
                    document = Document(file_path)
                    content = "\n".join([para.text for para in document.paragraphs])
                texts.append(content)
            except Exception as e:
                logging.warning(f"Could not load text from {file_path}: {e}")
                texts.append(f"Error loading {os.path.basename(file_path)}: {e}") # Add error message as content

        if not texts:
            logging.warning(f"No texts were successfully loaded from directory '{directory}'. Returning empty list.")
            return ([], 0)
            
        return (texts, len(texts))


class LoadImagesFromDirList:
    @classmethod
    def INPUT_TYPES(s):
        return (empty_pil_tensor(), empty_pil_tensor(channels=1).squeeze(-1), 0)


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
                "sort_criteria": (["filename", "newest_first", "oldest_first"], {"default": "filename"}),
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
            return hash(frozenset(kwargs.items()))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False, sort_criteria="filename"):
        if not os.path.isdir(directory):
            logging.error(f"Directory '{directory}' cannot be found. Returning empty lists.")
            return ([], [], [])

        try:
            dir_files_short = os.listdir(directory)
        except OSError as e:
            logging.error(f"Error listing directory '{directory}': {e}. Returning empty lists.")
            return ([], [], [])

        if not dir_files_short:
            logging.warning(f"No files in directory '{directory}'. Returning empty lists.")
            return ([], [], [])

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        if jxl:
            valid_extensions.extend('.jxl')

        processed_files = []
        for f_short in dir_files_short:
            if any(f_short.lower().endswith(ext) for ext in valid_extensions):
                full_path = os.path.join(directory, f_short)
                if not os.path.isdir(full_path): # Ensure it's a file
                    processed_files.append(full_path)

        if not processed_files:
            logging.warning(f"No valid image files found in directory '{directory}' after filtering. Returning empty lists.")
            return ([], [], [])

        # Sort files
        if sort_criteria == "newest_first":
            processed_files.sort(key=os.path.getmtime, reverse=True)
        elif sort_criteria == "oldest_first":
            processed_files.sort(key=os.path.getmtime)
        else:  # 'filename'
            processed_files.sort()

        # Apply start_index (note: LoadImagesFromDirList has min: 0 for start_index)
        if start_index < len(processed_files):
            files_to_process = processed_files[start_index:]
        else: # start_index is out of bounds
            files_to_process = []

        images = []
        masks = []
        file_paths = []

        # Cap the number of images to load
        if image_load_cap > 0:
            files_to_process = files_to_process[:image_load_cap]

        if not files_to_process:
            logging.warning(f"No images to load from '{directory}' after slicing/capping (start_index: {start_index}, cap: {image_load_cap}, sort: {sort_criteria}). Returning empty lists.")
            return ([], [], [])

        for image_path in files_to_process:
            try:
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                image_rgb = i.convert("RGB")
                image_np = np.array(image_rgb).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,] # (1, H, W, C)

                if 'A' in i.getbands():
                    mask_alpha = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    # Mask is (H, W), output list expects individual masks.
                    # For consistency with batched node, if it were to output single masks, they'd be (H,W)
                    mask_tensor = 1. - torch.from_numpy(mask_alpha) 
                else:
                    mask_tensor = torch.zeros((image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32, device="cpu")

                images.append(image_tensor)
                masks.append(mask_tensor) # Appending (H,W) mask
                file_paths.append(str(image_path))
            except Exception as e:
                logging.warning(f"Could not load image {image_path} for list: {e}")
                continue
        
        # If images list is empty after trying to load, it will correctly return ([], [], [])

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
            logging.warning(f"[Inspire Pack] ChangeImage(Latent)BatchSize: Unknown mode `{mode}` - ignored")
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
            logging.warning("[Inspire Pack] ColorMapToMasks - Sure, here's the translation: `color_map` can only be a single image. Only the first image will be processed. If you want to utilize the remaining images, convert the Image Batch to an Image List.")

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
    "LoadTextBatchFromDir //Inspire": LoadTextBatchFromDir,
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
    "LoadTextBatchFromDir //Inspire": "Load Text Batch From Dir (Inspire)",
    "SelectNthMask //Inspire": "Select Nth Mask (Inspire)"
}
