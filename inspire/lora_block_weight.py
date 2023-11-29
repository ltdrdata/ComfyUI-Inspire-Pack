import folder_paths
import comfy.utils
import comfy.lora
import os
import torch
import numpy as np
import nodes
import re

from server import PromptServer
from .libs import utils


def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def load_lbw_preset(filename):
    path = os.path.join(os.path.dirname(__file__), "..", "resources", filename)
    path = os.path.abspath(path)
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
        preset = [name for name in preset if not name.startswith('@')]

        lora_names = folder_paths.get_filename_list("loras")
        lora_dirs = [os.path.dirname(name) for name in lora_names]
        lora_dirs = ["All"] + list(set(lora_dirs))

        return {"required": {"model": ("MODEL",),
                             "clip": ("CLIP", ),
                             "category_filter": (lora_dirs,),
                             "lora_name": (lora_names, ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "inverse": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "A": ("FLOAT", {"default": 4.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "B": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "preset": (preset,),
                             "block_vector": ("STRING", {"multiline": True, "placeholder": "block weight vectors", "default": "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1", "pysssss.autocomplete": False}),
                             "bypass": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                             }
                }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "populated_vector")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/LoraBlockWeight"

    @staticmethod
    def validate(vectors):
        if len(vectors) < 12:
            return False

        for x in vectors:
            if x in ['R', 'r', 'U', 'u', 'A', 'a', 'B', 'b'] or is_numeric_string(x):
                continue
            else:
                subvectors = x.strip().split(' ')
                for y in subvectors:
                    y = y.strip()
                    if y not in ['R', 'r', 'U', 'u', 'A', 'a', 'B', 'b'] and not is_numeric_string(y):
                        return False

        return True

    @staticmethod
    def convert_vector_value(A, B, vector_value):
        def simple_vector(x):
            if x in ['U', 'u']:
                ratio = np.random.uniform(-1.5, 1.5)
                ratio = round(ratio, 2)
            elif x in ['R', 'r']:
                ratio = np.random.uniform(0, 3.0)
                ratio = round(ratio, 2)
            elif x == 'A':
                ratio = A
            elif x == 'a':
                ratio = A/2
            elif x == 'B':
                ratio = B
            elif x == 'b':
                ratio = B/2
            elif is_numeric_string(x):
                ratio = float(x)
            else:
                ratio = None

            return ratio

        v = simple_vector(vector_value)
        if v is not None:
            ratios = [v]
        else:
            ratios = [simple_vector(x) for x in vector_value.split(" ")]

        return ratios

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
            preset_dict = load_preset_dict()
            if len(vector) > 0 and vector[0].strip() in preset_dict:
                vector = preset_dict[vector[0].strip()].split(",")
            else:
                raise ValueError(f"[LoraLoaderBlockWeight] invalid block_vector '{block_vector}'")

        last_k_unet_num = None
        new_modelpatcher = model.clone()
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
        ratios = []
        for k, v, k_unet_num, k_unet in (input_blocks + middle_blocks + output_blocks):
            if last_k_unet_num != k_unet_num and len(vector) > vector_i:
                ratios = LoraLoaderBlockWeight.convert_vector_value(A, B, vector[vector_i].strip())
                ratio = ratios.pop(0)

                if inverse:
                    populated_ratio = 1 - ratio
                else:
                    populated_ratio = ratio

                populated_vector_list.append(LoraLoaderBlockWeight.norm_value(populated_ratio))

                vector_i += 1
            else:
                if len(ratios) > 0:
                    ratio = ratios.pop(0)

                if inverse:
                    populated_ratio = 1 - ratio
                else:
                    populated_ratio = ratio

            last_k_unet_num = k_unet_num

            new_modelpatcher.add_patches({k: v}, strength_model * populated_ratio)
            # if inverse:
            #     print(f"\t{k_unet} -> inv({ratio}) ")
            # else:
            #     print(f"\t{k_unet} -> ({ratio}) ")

        # prepare base patch
        ratios = LoraLoaderBlockWeight.convert_vector_value(A, B, vector[0].strip())
        ratio = ratios.pop(0)

        if inverse:
            populated_ratio = 1 - ratio

        populated_vector_list.insert(0, LoraLoaderBlockWeight.norm_value(populated_ratio))

        for k, v, k_unet in others:
            new_modelpatcher.add_patches({k: v}, strength_model * populated_ratio)
            # if inverse:
            #     print(f"\t{k_unet} -> inv({ratio}) ")
            # else:
            #     print(f"\t{k_unet} -> ({ratio}) ")

        new_clip = clip.clone()
        new_clip.add_patches(loaded, strength_clip)
        populated_vector = ','.join(map(str, populated_vector_list))
        return (new_modelpatcher, new_clip, populated_vector)

    def doit(self, model, clip, lora_name, strength_model, strength_clip, inverse, seed, A, B, preset, block_vector, bypass=False, category_filter=None):
        if strength_model == 0 and strength_clip == 0 or bypass:
            return (model, clip, "")

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
            print(f"XY_Capsule_LoraBlockWeight: ({self.another_capsule.x, self.y}) is processed.")
            self.storage[(self.another_capsule.x, self.y)] = image
        else:
            print(f"XY_Capsule_LoraBlockWeight: ({self.x, self.y}) is processed.")

    def patch_model(self, model, clip):
        lora_name, strength_model, strength_clip, inverse, block_vectors, seed, A, B, heatmap_palette, heatmap_alpha, heatmap_strength, xyplot_mode = self.params

        try:
            if self.y == 0:
                target_vector = self.another_capsule.target_vector if self.another_capsule else self.target_vector
                model, clip, _ = LoraLoaderBlockWeight().doit(model, clip, lora_name, strength_model, strength_clip, inverse,
                                                              seed, A, B, "", target_vector)
            elif self.y == 1:
                reference_vector = self.another_capsule.reference_vector if self.another_capsule else self.reference_vector
                model, clip, _ = LoraLoaderBlockWeight().doit(model, clip, lora_name, strength_model, strength_clip, inverse,
                                                              seed, A, B, "", reference_vector)
        except:
            self.storage[(self.another_capsule.x, self.y)] = "fail"
            pass

        return model, clip

    def pre_define_model(self, model, clip, vae):
        if self.y < 2:
            model, clip = self.patch_model(model, clip)

        return model, clip, vae

    def get_result(self, model, clip, vae):
        _, _, _, _, _, _, _, _, heatmap_palette, heatmap_alpha, heatmap_strength, xyplot_mode = self.params

        if self.y < 2:
            return None

        if self.y == 2:
            # diff
            weighted_image = self.storage[(self.another_capsule.x, 0)]
            reference_image = self.storage[(self.another_capsule.x, 1)]

            if weighted_image == "fail" or reference_image == "fail":
                image = "fail"
            else:
                image = torch.abs(weighted_image - reference_image)
                self.storage[(self.another_capsule.x, self.y)] = image
        elif self.y == 3:
            import matplotlib.cm as cm
            # heatmap
            image = self.storage[(self.another_capsule.x, 0)]

            if image == "fail":
                image = utils.empty_pil_tensor(8,8)
                latent = utils.empty_latent()
                return (image, latent)
            else:
                image = image.clone()

                diff_image = torch.abs(self.storage[(self.another_capsule.x, 2)])

                heatmap = torch.sum(diff_image, dim=3, keepdim=True)

                min_val = torch.min(heatmap)
                max_val = torch.max(heatmap)
                heatmap = (heatmap - min_val) / (max_val - min_val)
                heatmap *= heatmap_strength

                # viridis / magma / plasma / inferno / cividis
                if heatmap_palette == "magma":
                    colormap = cm.magma
                elif heatmap_palette == "plasma":
                    colormap = cm.plasma
                elif heatmap_palette == "inferno":
                    colormap = cm.inferno
                elif heatmap_palette == "cividis":
                    colormap = cm.cividis
                else:
                    # default: viridis
                    colormap = cm.viridis

                heatmap = torch.from_numpy(colormap(heatmap.squeeze())).unsqueeze(0)
                heatmap = heatmap[..., :3]

                image = heatmap_alpha * heatmap + (1 - heatmap_alpha) * image

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
        if not x.startswith('@'):
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

        lora_names = folder_paths.get_filename_list("loras")
        lora_dirs = [os.path.dirname(name) for name in lora_names]
        lora_dirs = ["All"] + list(set(lora_dirs))

        return {"required": {
                             "category_filter": (lora_dirs, ),
                             "lora_name": (lora_names, ),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "inverse": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "A": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "B": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "preset": (preset,),
                             "block_vectors": ("STRING", {"multiline": True, "default": default_vectors, "placeholder": "{target vector}/{reference vector}", "pysssss.autocomplete": False}),
                             "heatmap_palette": (["viridis", "magma", "plasma", "inferno", "cividis"], ),
                             "heatmap_alpha":  ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "heatmap_strength": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "xyplot_mode": (["Simple", "Diff", "Diff+Heatmap"],),
                             }}

    RETURN_TYPES = ("XY", "XY")
    RETURN_NAMES = ("X (vectors)", "Y (effect_compares)")

    FUNCTION = "doit"
    CATEGORY = "InspirePack/LoraBlockWeight"

    def doit(self, lora_name, strength_model, strength_clip, inverse, seed, A, B, preset, block_vectors, heatmap_palette, heatmap_alpha, heatmap_strength, xyplot_mode, category_filter=None):
        xy_type = "XY_Capsule"

        preset_dict = load_preset_dict()
        common_params = lora_name, strength_model, strength_clip, inverse, block_vectors, seed, A, B, heatmap_palette, heatmap_alpha, heatmap_strength, xyplot_mode

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
                    x_item = XY_Capsule_LoraBlockWeight(x_idx, 0, block_vector, label, storage, common_params)
                    x_idx += 1

                if x_item is not None and ref_block_vector is not None:
                    x_item.set_reference_vector(ref_block_vector)

                if x_item is not None:
                    x_values.append(x_item)

        if xyplot_mode == "Simple":
            y_values = [XY_Capsule_LoraBlockWeight(0, 0, '', 'target', storage, common_params)]
        elif xyplot_mode == "Diff":
            y_values = [XY_Capsule_LoraBlockWeight(0, 0, '', 'target', storage, common_params),
                        XY_Capsule_LoraBlockWeight(0, 1, '', 'reference', storage, common_params),
                        XY_Capsule_LoraBlockWeight(0, 2, '', 'diff', storage, common_params)]
        else:
            y_values = [XY_Capsule_LoraBlockWeight(0, 0, '', 'target', storage, common_params),
                        XY_Capsule_LoraBlockWeight(0, 1, '', 'reference', storage, common_params),
                        XY_Capsule_LoraBlockWeight(0, 2, '', 'diff', storage, common_params),
                        XY_Capsule_LoraBlockWeight(0, 3, '', 'heatmap', storage, common_params)]

        return ((xy_type, x_values), (xy_type, y_values), )


class LoraBlockInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "model": ("MODEL", ),
                        "clip": ("CLIP", ),
                        "lora_name": (folder_paths.get_filename_list("loras"), ),
                        "block_info": ("STRING", {"multiline": True}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    CATEGORY = "InspirePack/LoraBlockWeight"

    OUTPUT_NODE = True

    RETURN_TYPES = ()
    FUNCTION = "doit"

    @staticmethod
    def extract_info(model, clip, lora):
        key_map = comfy.lora.model_lora_keys_unet(model.model)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        loaded = comfy.lora.load_lora(lora, key_map)

        def parse_unet_num(s):
            if s[1] == '.':
                return int(s[0])
            else:
                return int(s)

        input_block_count = set()
        input_blocks = []
        input_blocks_map = {}

        middle_block_count = set()
        middle_blocks = []
        middle_blocks_map = {}

        output_block_count = set()
        output_blocks = []
        output_blocks_map = {}

        text_block_count = set()
        text_blocks = []
        text_blocks_map = {}

        others = []
        for k, v in loaded.items():
            k_unet = k[len("diffusion_model."):]

            if k_unet.startswith("input_blocks."):
                k_unet_num = k_unet[len("input_blocks."):len("input_blocks.")+2]
                k_unet_int = parse_unet_num(k_unet_num)

                input_block_count.add(k_unet_int)
                input_blocks.append(k_unet)
                if k_unet_int in input_blocks_map:
                    input_blocks_map[k_unet_int].append(k_unet)
                else:
                    input_blocks_map[k_unet_int] = [k_unet]

            elif k_unet.startswith("middle_block."):
                k_unet_num = k_unet[len("middle_block."):len("middle_block.")+2]
                k_unet_int = parse_unet_num(k_unet_num)

                middle_block_count.add(k_unet_int)
                middle_blocks.append(k_unet)
                if k_unet_int in middle_blocks_map:
                    middle_blocks_map[k_unet_int].append(k_unet)
                else:
                    middle_blocks_map[k_unet_int] = [k_unet]

            elif k_unet.startswith("output_blocks."):
                k_unet_num = k_unet[len("output_blocks."):len("output_blocks.")+2]
                k_unet_int = parse_unet_num(k_unet_num)

                output_block_count.add(k_unet_int)
                output_blocks.append(k_unet)
                if k_unet_int in output_blocks_map:
                    output_blocks_map[k_unet_int].append(k_unet)
                else:
                    output_blocks_map[k_unet_int] = [k_unet]

            elif k_unet.startswith("_model.encoder.layers."):
                k_unet_num = k_unet[len("_model.encoder.layers."):len("_model.encoder.layers.")+2]
                k_unet_int = parse_unet_num(k_unet_num)

                text_block_count.add(k_unet_int)
                text_blocks.append(k_unet)
                if k_unet_int in text_blocks_map:
                    text_blocks_map[k_unet_int].append(k_unet)
                else:
                    text_blocks_map[k_unet_int] = [k_unet]

            else:
                others.append(k_unet)

        text = ""

        input_blocks = sorted(input_blocks)
        middle_blocks = sorted(middle_blocks)
        output_blocks = sorted(output_blocks)
        others = sorted(others)

        text += f"\n-------[Input blocks] ({len(input_block_count)}, Subs={len(input_blocks)})-------\n"
        input_keys = sorted(input_blocks_map.keys())
        for x in input_keys:
            text += f" IN{x}: {len(input_blocks_map[x])}\n"

        text += f"\n-------[Middle blocks] ({len(middle_block_count)}, Subs={len(middle_blocks)})-------\n"
        middle_keys = sorted(middle_blocks_map.keys())
        for x in middle_keys:
            text += f" MID{x}: {len(middle_blocks_map[x])}\n"

        text += f"\n-------[Output blocks] ({len(output_block_count)}, Subs={len(output_blocks)})-------\n"
        output_keys = sorted(output_blocks_map.keys())
        for x in output_keys:
            text += f" OUT{x}: {len(output_blocks_map[x])}\n"

        text += f"\n-------[Text blocks] ({len(text_block_count)}, Subs={len(text_blocks)})-------\n"
        text_keys = sorted(text_blocks_map.keys())
        for x in text_keys:
            text += f" CLIP{x}: {len(text_blocks_map[x])}\n"

        text += f"\n-------[Base blocks] ({len(others)})-------\n"
        for x in others:
            text += f" {x}\n"

        return text

    def doit(self, model, clip, lora_name, block_info, unique_id):
        lora_path = folder_paths.get_full_path("loras", lora_name)

        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        text = LoraBlockInfo.extract_info(model, clip, lora)

        PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": unique_id, "widget_name": "block_info", "type": "text", "data": text})
        return {}


NODE_CLASS_MAPPINGS = {
    "XY Input: Lora Block Weight //Inspire": XYInput_LoraBlockWeight,
    "LoraLoaderBlockWeight //Inspire": LoraLoaderBlockWeight,
    "LoraBlockInfo //Inspire": LoraBlockInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "XY Input: Lora Block Weight //Inspire": "XY Input: Lora Block Weight",
    "LoraLoaderBlockWeight //Inspire": "Lora Loader (Block Weight)",
    "LoraBlockInfo //Inspire": "Lora Block Info",
}
