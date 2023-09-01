import folder_paths
import comfy.utils
import comfy.lora
import os
import torch
import numpy as np
import nodes
from scipy.ndimage import gaussian_filter
import re


def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def load_lbw_preset(filename):
    path = os.path.join(os.path.dirname(__file__), "resources", filename)
    preset_list = []

    if os.path.exists(path):
        with open(path, 'r') as file:
            for line in file:
                preset_list.append(line.strip())

        return preset_list
    else:
        return []


class LoraLoaderBlockWeight:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        preset = ["Preset"]  # 20
        preset += load_lbw_preset("lbw-preset.txt")
        preset += load_lbw_preset("lbw-preset.custom.txt")

        return {"required": {"model": ("MODEL",),
                             "clip": ("CLIP", ),
                             "lora_name": (folder_paths.get_filename_list("loras"), ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "inverse": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "A": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "B": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "preset": (preset,),
                             "block_vector": ("STRING", {"multiline": True, "placeholder": "block weight vectors", "default": "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1"}),
                              }}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "populated_vector")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/LoraBlockWeight"

    @staticmethod
    def validate(vectors):
        if len(vectors) < 12:
            return False

        for x in vectors:
            if x not in ['R', 'r', 'U', 'u', 'A', 'a', 'B', 'b'] and not is_numeric_string(x):
                return False

        return True

    @staticmethod
    def convert_vector_value(A, B, vector_value):
        if vector_value in ['U', 'u']:
            ratio = np.random.uniform(-1.5, 1.5)
            ratio = round(ratio, 2)
        elif vector_value in ['R', 'r']:
            ratio = np.random.uniform(0, 3.0)
            ratio = round(ratio, 2)
        elif vector_value == 'A':
            ratio = A
        elif vector_value == 'a':
            ratio = A/2
        elif vector_value == 'B':
            ratio = B
        elif vector_value == 'b':
            ratio = B/2
        else:
            ratio = float(vector_value)

        return ratio

    @staticmethod
    def norm_value(value):  # make to int if 1.0 or 0.0
        if value == 1:
            return 1
        elif value == 0:
            return 0
        else:
            return value

    @staticmethod
    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, inverse, seed, A, B, block_vector):
        key_map = comfy.lora.model_lora_keys_unet(model.model)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        loaded = comfy.lora.load_lora(lora, key_map)

        block_vector = block_vector.split(":")
        if len(block_vector) > 1:
            block_vector = block_vector[1]
        else:
            block_vector = block_vector[0]

        vector = block_vector.split(",")
        vector_i = 1

        if not LoraLoaderBlockWeight.validate(vector):
            raise ValueError(f"[LoraLoaderBlockWeight] invalid block_vector '{block_vector}'")


        last_k_unet_num = None
        new_modelpatcher = model.clone()
        ratio = strength_model
        populated_ratio = strength_model

        def parse_unet_num(s):
            if s[1] == '.':
                return int(s[0])
            else:
                return int(s)

        # sort: input, middle, output, others
        input_blocks = []
        middle_blocks = []
        output_blocks = []
        others = []
        for k, v in loaded.items():
            k_unet = k[len("diffusion_model."):]

            if k_unet.startswith("input_blocks."):
                k_unet_num = k_unet[len("input_blocks."):len("input_blocks.")+2]
                input_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
            elif k_unet.startswith("middle_block."):
                k_unet_num = k_unet[len("middle_block."):len("middle_block.")+2]
                middle_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
            elif k_unet.startswith("output_blocks."):
                k_unet_num = k_unet[len("output_blocks."):len("output_blocks.")+2]
                output_blocks.append((k, v, parse_unet_num(k_unet_num), k_unet))
            else:
                others.append((k, v, k_unet))

        input_blocks = sorted(input_blocks, key=lambda x: x[2])
        middle_blocks = sorted(middle_blocks, key=lambda x: x[2])
        output_blocks = sorted(output_blocks, key=lambda x: x[2])

        # prepare patch
        np.random.seed(seed % (2**31))
        populated_vector_list = []
        for k, v, k_unet_num, k_unet in (input_blocks + middle_blocks + output_blocks):
            if last_k_unet_num != k_unet_num and len(vector) > vector_i:
                ratio = LoraLoaderBlockWeight.convert_vector_value(A, B, vector[vector_i].strip())

                if inverse:
                    populated_ratio = 1 - ratio
                else:
                    populated_ratio = ratio

                populated_vector_list.append(LoraLoaderBlockWeight.norm_value(populated_ratio))

                vector_i += 1

            last_k_unet_num = k_unet_num

            new_modelpatcher.add_patches({k: v}, strength_model * populated_ratio)
            if inverse:
                print(f"\t{k_unet} -> inv({ratio}) ")
            else:
                print(f"\t{k_unet} -> ({ratio}) ")

        # prepare base patch
        ratio = LoraLoaderBlockWeight.convert_vector_value(A, B, vector[0].strip())

        if inverse:
            populated_ratio = 1 - ratio
        else:
            populated_ratio = ratio

        populated_vector_list.insert(0, LoraLoaderBlockWeight.norm_value(populated_ratio))

        for k, v, k_unet in others:
            new_modelpatcher.add_patches({k: v}, strength_model * populated_ratio)
            if inverse:
                print(f"\t{k_unet} -> inv({ratio}) ")
            else:
                print(f"\t{k_unet} -> ({ratio}) ")

        new_clip = clip.clone()
        new_clip.add_patches(loaded, strength_clip)
        populated_vector = ','.join(map(str, populated_vector_list))
        return (new_modelpatcher, new_clip, populated_vector)

    def doit(self, model, clip, lora_name, strength_model, strength_clip, inverse, seed, A, B, preset, block_vector):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora, populated_vector = LoraLoaderBlockWeight.load_lora_for_models(model, clip, lora, strength_model, strength_clip, inverse, seed, A, B, block_vector)
        return (model_lora, clip_lora, populated_vector)


class XY_Capsule_LoraBlockWeight:
    def __init__(self, x, y, target_vector, label, storage, params):
        self.x = x
        self.y = y
        self.target_vector = target_vector
        self.reference_vector = None
        self.label = label
        self.storage = storage
        self.another_capsule = None
        self.params = params

    def set_reference_vector(self, vector):
        self.reference_vector = vector

    def set_x_capsule(self, capsule):
        self.another_capsule = capsule

    def set_result(self, image, latent):
        if self.another_capsule is not None:
            self.storage[(self.another_capsule.x, self.y)] = image

    def patch_model(self, model, clip):
        lora_name, strength_model, strength_clip, inverse, block_vectors, seed, A, B = self.params
        if self.y == 0:
            target_vector = self.another_capsule.target_vector if self.another_capsule else self.target_vector
            model, clip, _ = LoraLoaderBlockWeight().doit(model, clip, lora_name, strength_model, strength_clip, inverse,
                                                          seed, A, B, "", target_vector)
        elif self.y == 1:
            reference_vector = self.another_capsule.reference_vector if self.another_capsule else self.reference_vector
            model, clip, _ = LoraLoaderBlockWeight().doit(model, clip, lora_name, strength_model, strength_clip, inverse,
                                                          seed, A, B, "", reference_vector)

        return model, clip

    def pre_define_model(self, model, clip, vae):
        if self.y < 2:
            model, clip = self.patch_model(model, clip)

        return model, clip, vae

    def get_result(self, model, clip, vae):
        if self.y < 2:
            return None

        if self.y == 2:
            # diff
            weighted_image = self.storage[(self.another_capsule.x, 0)]
            reference_image = self.storage[(self.another_capsule.x, 1)]
            image = torch.abs(weighted_image - reference_image)
            self.storage[(self.another_capsule.x, self.y)] = image
        elif self.y == 3:
            # heatmap
            heatmap = self.storage[(self.another_capsule.x, 2)].squeeze().sum(dim=-1)
            image = self.storage[(self.another_capsule.x, 0)].clone()

            heatmap = heatmap.unsqueeze(0).unsqueeze(-1)
            heatmap = heatmap.expand(image.shape)

            # Create a yellow mask based on the blurred heatmap
            yellow_mask = torch.zeros_like(image)
            yellow_mask[..., 0] = 1.0  # Set red channel to 1 (full intensity)
            yellow_mask[..., 1] = 1.0  # Set green channel to 1 (full intensity)

            # Combine the yellow mask with the blurred heatmap
            combined_image = image + 0.6 * heatmap * yellow_mask

            # Make sure values are within the valid range [0, 1]
            image = torch.clamp(combined_image, 0, 1)

        latent = nodes.VAEEncode().encode(vae, image)[0]
        return (image, latent)

    def getLabel(self):
        return self.label


def load_preset_dict():
    preset = ["Preset"]  # 20
    preset += load_lbw_preset("lbw-preset.txt")
    preset += load_lbw_preset("lbw-preset.custom.txt")

    dict = {}
    for x in preset:
        item = x.split(':')
        if len(item) > 1:
            dict[item[0]] = item[1]

    return dict


class XYInput_LoraBlockWeight:
    @staticmethod
    def resolve_vector_string(vector_string, preset_dict):
        vector_string = vector_string.strip()

        if vector_string in preset_dict:
            return vector_string, preset_dict[vector_string]

        vector_infos = vector_string.split(':')

        if len(vector_infos) > 1:
            return vector_infos[0], vector_infos[1]
        elif len(vector_infos) > 0:
            return vector_infos[0], vector_infos[0]
        else:
            return None, None

    @classmethod
    def INPUT_TYPES(cls):
        preset = ["Preset"]  # 20
        preset += load_lbw_preset("lbw-preset.txt")
        preset += load_lbw_preset("lbw-preset.custom.txt")

        default_vectors = "SD-NONE/SD-ALL\nSD-ALL/SD-ALL\nSD-INS/SD-ALL\nSD-IND/SD-ALL\nSD-INALL/SD-ALL\nSD-MIDD/SD-ALL\nSD-MIDD0.2/SD-ALL\nSD-MIDD0.8/SD-ALL\nSD-MOUT/SD-ALL\nSD-OUTD/SD-ALL\nSD-OUTS/SD-ALL\nSD-OUTALL/SD-ALL"

        return {"required": {"lora_name": (folder_paths.get_filename_list("loras"), ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "inverse": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "A": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "B": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "preset": (preset,),
                             "block_vectors": ("STRING", {"multiline": True, "default": default_vectors, "placeholder": "{target vector}/{reference vector}"}),
                             }}

    RETURN_TYPES = ("XY", "XY")
    RETURN_NAMES = ("X (vectors)", "Y (effect_compares)")

    FUNCTION = "doit"
    CATEGORY = "InspirePack/LoraBlockWeight"

    def doit(self, lora_name, strength_model, strength_clip, inverse, seed, A, B, preset, block_vectors):
        xy_type = "XY_Capsule"

        preset_dict = load_preset_dict()
        lora_params = lora_name, strength_model, strength_clip, inverse, block_vectors, seed, A, B

        storage = {}
        x_values = []
        x_idx = 0
        for block_vector in block_vectors.split("\n"):
            if block_vector == "":
                continue

            item = block_vector.split('/')

            if len(item) > 0:
                target_vector = item[0].strip()
                ref_vector = item[1].strip() if len(item) > 1 else ''

                x_item = None
                label, block_vector = XYInput_LoraBlockWeight.resolve_vector_string(target_vector, preset_dict)
                _, ref_block_vector = XYInput_LoraBlockWeight.resolve_vector_string(ref_vector, preset_dict)
                if label is not None:
                    x_item = XY_Capsule_LoraBlockWeight(x_idx, 0, block_vector, label, storage, lora_params)
                    x_idx += 1

                if x_item is not None and ref_block_vector is not None:
                    x_item.set_reference_vector(ref_block_vector)

                if x_item is not None:
                    x_values.append(x_item)

        y_values = [XY_Capsule_LoraBlockWeight(0, 0, '', 'target', storage, lora_params),
                    XY_Capsule_LoraBlockWeight(0, 1, '', 'reference', storage, lora_params),
                    XY_Capsule_LoraBlockWeight(0, 2, '', 'diff', storage, lora_params),
                    XY_Capsule_LoraBlockWeight(0, 3, '', 'heatmap', storage, lora_params)]

        return ((xy_type, x_values), (xy_type, y_values), )


NODE_CLASS_MAPPINGS = {
    "XY Input: Lora Block Weight //Inspire": XYInput_LoraBlockWeight,
    "LoraLoaderBlockWeight //Inspire": LoraLoaderBlockWeight
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "XY Input: Lora Block Weight //Inspire": "XY Input: Lora Block Weight",
    "LoraLoaderBlockWeight //Inspire": "Lora Loader (Block Weight)"
}
