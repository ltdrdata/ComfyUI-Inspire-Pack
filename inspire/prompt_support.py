import os
import re
import json
import sys
import shutil
import yaml

from PIL import Image
import nodes
import torch

import folder_paths
import comfy
import traceback
import random

from server import PromptServer
from .libs import utils
from .backend_support import CheckpointLoaderSimpleShared

prompt_builder_preset = {}


resource_path = os.path.join(os.path.dirname(__file__), "..", "resources")
resource_path = os.path.abspath(resource_path)

prompts_path = os.path.join(os.path.dirname(__file__), "..", "prompts")
prompts_path = os.path.abspath(prompts_path)


try:
    pb_yaml_path = os.path.join(resource_path, 'prompt-builder.yaml')
    pb_yaml_path_example = os.path.join(resource_path, 'prompt-builder.yaml.example')

    if not os.path.exists(pb_yaml_path):
        shutil.copy(pb_yaml_path_example, pb_yaml_path)

    with open(pb_yaml_path, 'r', encoding="utf-8") as f:
        prompt_builder_preset = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    print(f"[Inspire Pack] Failed to load 'prompt-builder.yaml'")


class LoadPromptsFromDir:
    @classmethod
    def INPUT_TYPES(cls):
        global prompts_path
        try:
            prompt_dirs = [d for d in os.listdir(prompts_path) if os.path.isdir(os.path.join(prompts_path, d))]
        except Exception:
            prompt_dirs = []

        return {"required": {"prompt_dir": (prompt_dirs,)}}

    RETURN_TYPES = ("ZIPPED_PROMPT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    def doit(self, prompt_dir):
        global prompts_path
        prompt_dir = os.path.join(prompts_path, prompt_dir)
        files = [f for f in os.listdir(prompt_dir) if f.endswith(".txt")]
        files.sort()

        prompts = []
        for file_name in files:
            print(f"file_name: {file_name}")
            try:
                with open(os.path.join(prompt_dir, file_name), "r", encoding="utf-8") as file:
                    prompt_data = file.read()
                    prompt_list = re.split(r'\n\s*-+\s*\n', prompt_data)

                    for prompt in prompt_list:
                        pattern = r"positive:(.*?)(?:\n*|$)negative:(.*)"
                        matches = re.search(pattern, prompt, re.DOTALL)

                        if matches:
                            positive_text = matches.group(1).strip()
                            negative_text = matches.group(2).strip()
                            result_tuple = (positive_text, negative_text, file_name)
                            prompts.append(result_tuple)
                        else:
                            print(f"[WARN] LoadPromptsFromDir: invalid prompt format in '{file_name}'")
            except Exception as e:
                print(f"[ERROR] LoadPromptsFromDir: an error occurred while processing '{file_name}': {str(e)}")

        return (prompts, )


class LoadPromptsFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        global prompts_path
        try:
            prompt_files = []
            for root, dirs, files in os.walk(prompts_path):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, prompts_path)
                        prompt_files.append(rel_path)
        except Exception:
            prompt_files = []

        return {"required": {"prompt_file": (prompt_files,)}}

    RETURN_TYPES = ("ZIPPED_PROMPT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    def doit(self, prompt_file):
        prompt_path = os.path.join(prompts_path, prompt_file)

        prompts = []
        try:
            with open(prompt_path, "r", encoding="utf-8") as file:
                prompt_data = file.read()
                prompt_list = re.split(r'\n\s*-+\s*\n', prompt_data)

                pattern = r"positive:(.*?)(?:\n*|$)negative:(.*)"

                for prompt in prompt_list:
                    matches = re.search(pattern, prompt, re.DOTALL)

                    if matches:
                        positive_text = matches.group(1).strip()
                        negative_text = matches.group(2).strip()
                        result_tuple = (positive_text, negative_text, prompt_file)
                        prompts.append(result_tuple)
                    else:
                        print(f"[WARN] LoadPromptsFromFile: invalid prompt format in '{prompt_file}'")
        except Exception as e:
            print(f"[ERROR] LoadPromptsFromFile: an error occurred while processing '{prompt_file}': {str(e)}")

        return (prompts, )


class LoadSinglePromptFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        global prompts_path
        try:
            prompt_files = []
            for root, dirs, files in os.walk(prompts_path):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, prompts_path)
                        prompt_files.append(rel_path)
        except Exception:
            prompt_files = []

        return {"required": {
            "prompt_file": (prompt_files,),
            "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("ZIPPED_PROMPT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    def doit(self, prompt_file, index):
        prompt_path = os.path.join(prompts_path, prompt_file)

        prompts = []
        try:
            with open(prompt_path, "r", encoding="utf-8") as file:
                prompt_data = file.read()
                prompt_list = re.split(r'\n\s*-+\s*\n', prompt_data)
                try:
                    prompt = prompt_list[index]
                except Exception:
                    prompt = prompt_list[-1]

                pattern = r"positive:(.*?)(?:\n*|$)negative:(.*)"
                matches = re.search(pattern, prompt, re.DOTALL)

                if matches:
                    positive_text = matches.group(1).strip()
                    negative_text = matches.group(2).strip()
                    result_tuple = (positive_text, negative_text, prompt_file)
                    prompts.append(result_tuple)
                else:
                    print(f"[WARN] LoadPromptsFromFile: invalid prompt format in '{prompt_file}'")
        except Exception as e:
            print(f"[ERROR] LoadPromptsFromFile: an error occurred while processing '{prompt_file}': {str(e)}")

        return (prompts, )


class UnzipPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"zipped_prompt": ("ZIPPED_PROMPT",), }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "name")

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    def doit(self, zipped_prompt):
        return zipped_prompt


class ZipPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "positive": ("STRING", {"forceInput": True, "multiline": True}),
                    "negative": ("STRING", {"forceInput": True, "multiline": True}),
                    },
                "optional": {
                    "name_opt": ("STRING", {"forceInput": True, "multiline": False})
                    }
                }

    RETURN_TYPES = ("ZIPPED_PROMPT",)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    def doit(self, positive, negative, name_opt=""):
        return ((positive, negative, name_opt), )


prompt_blacklist = set([
    'filename_prefix'
])

class PromptExtractor:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {
                    "image": (sorted(files), {"image_upload": True}),
                    "positive_id": ("STRING", {}),
                    "negative_id": ("STRING", {}),
                    "info": ("STRING", {"multiline": True})
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    CATEGORY = "InspirePack/Prompt"

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "doit"

    OUTPUT_NODE = True

    def doit(self, image, positive_id, negative_id, info, unique_id):
        image_path = folder_paths.get_annotated_filepath(image)
        info = Image.open(image_path).info

        positive = ""
        negative = ""
        text = ""
        prompt_dicts = {}
        node_inputs = {}

        def get_node_inputs(x):
            if x in node_inputs:
                return node_inputs[x]
            else:
                node_inputs[x] = None

                obj = nodes.NODE_CLASS_MAPPINGS.get(x, None)
                if obj is not None:
                    input_types = obj.INPUT_TYPES()
                    node_inputs[x] = input_types
                    return input_types
                else:
                    return None

        if isinstance(info, dict) and 'workflow' in info:
            prompt = json.loads(info['prompt'])
            for k, v in prompt.items():
                input_types = get_node_inputs(v['class_type'])
                if input_types is not None:
                    inputs = input_types['required'].copy()
                    if 'optional' in input_types:
                        inputs.update(input_types['optional'])

                    for name, value in inputs.items():
                        if name in prompt_blacklist:
                            continue

                        if value[0] == 'STRING' and name in v['inputs']:
                            prompt_dicts[f"{k}.{name.strip()}"] = (v['class_type'], v['inputs'][name])

            for k, v in prompt_dicts.items():
                text += f"{k} [{v[0]}] ==> {v[1]}\n"

            positive = prompt_dicts.get(positive_id.strip(), "")
            negative = prompt_dicts.get(negative_id.strip(), "")
        else:
            text = "There is no prompt information within the image."

        PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": unique_id, "widget_name": "info", "type": "text", "data": text})
        return (positive, negative)


class GlobalSeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "control_before_generate", "label_off": "control_after_generate"}),
                "action": (["fixed", "increment", "decrement", "randomize",
                            "increment for each node", "decrement for each node", "randomize for each node"], ),
                "last_seed": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    OUTPUT_NODE = True

    def doit(self, **kwargs):
        return {}


class GlobalSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    OUTPUT_NODE = True

    def doit(self, **kwargs):
        return {}


class BindImageListPromptList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "zipped_prompts": ("ZIPPED_PROMPT",),
                "default_positive": ("STRING", {"multiline": True, "placeholder": "default positive"}),
                "default_negative": ("STRING", {"multiline": True, "placeholder": "default negative"}),
            }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "positive", "negative", "prompt_label")

    OUTPUT_IS_LIST = (True, True, True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    def doit(self, images, zipped_prompts, default_positive, default_negative):
        positives = []
        negatives = []
        prompt_labels = []

        if len(images) < len(zipped_prompts):
            zipped_prompts = zipped_prompts[:len(images)]

        elif len(images) > len(zipped_prompts):
            lack = len(images) - len(zipped_prompts)
            default_prompt = (default_positive[0], default_negative[0], "default")
            zipped_prompts = zipped_prompts[:]
            for i in range(lack):
                zipped_prompts.append(default_prompt)

        for prompt in zipped_prompts:
            a, b, c = prompt
            positives.append(a)
            negatives.append(b)
            prompt_labels.append(c)

        return (images, positives, negatives, prompt_labels)


class BNK_EncoderWrapper:
    def __init__(self, token_normalization, weight_interpretation):
        self.token_normalization = token_normalization
        self.weight_interpretation = weight_interpretation

    def encode(self, clip, text):
        if 'BNK_CLIPTextEncodeAdvanced' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb',
                                          "To use 'WildcardEncodeInspire' node, 'ComfyUI_ADV_CLIP_emb' extension is required.")
            raise Exception(f"[ERROR] To use WildcardEncodeInspire, you need to install 'Advanced CLIP Text Encode'")
        return nodes.NODE_CLASS_MAPPINGS['BNK_CLIPTextEncodeAdvanced']().encode(clip, text, self.token_normalization, self.weight_interpretation)


class WildcardEncodeInspire:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "model": ("MODEL",),
                        "clip": ("CLIP",),
                        "token_normalization": (["none", "mean", "length", "length+mean"], ),
                        "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"], {'default': 'comfy++'}),
                        "wildcard_text": ("STRING", {"multiline": True, "dynamicPrompts": False, 'placeholder': 'Wildcard Prompt (User Input)'}),
                        "populated_text": ("STRING", {"multiline": True, "dynamicPrompts": False, 'placeholder': 'Populated Prompt (Will be generated automatically)'}),
                        "mode": ("BOOLEAN", {"default": True, "label_on": "Populate", "label_off": "Fixed"}),
                        "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"), ),
                        "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                }

    CATEGORY = "InspirePack/Prompt"

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "clip", "conditioning", "populated_text")
    FUNCTION = "doit"

    def doit(self, *args, **kwargs):
        populated = kwargs['populated_text']

        clip_encoder = BNK_EncoderWrapper(kwargs['token_normalization'], kwargs['weight_interpretation'])

        if 'ImpactWildcardEncode' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/ltdrdata/ComfyUI-Impact-Pack',
                                          "To use 'Wildcard Encode (Inspire)' node, 'Impact Pack' extension is required.")
            raise Exception(f"[ERROR] To use 'Wildcard Encode (Inspire)', you need to install 'Impact Pack'")

        model, clip, conditioning = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardEncode'].process_with_loras(wildcard_opt=populated, model=kwargs['model'], clip=kwargs['clip'], clip_encoder=clip_encoder)
        return (model, clip, conditioning, populated)


class MakeBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "ckpt_key_opt": ("STRING", {"multiline": False, "placeholder": "If empty, use 'ckpt_name' as the key." }),

                        "positive_wildcard_text": ("STRING", {"multiline": True, "dynamicPrompts": False, 'placeholder': 'Positive Prompt (User Input)'}),
                        "negative_wildcard_text": ("STRING", {"multiline": True, "dynamicPrompts": False, 'placeholder': 'Negative Prompt (User Input)'}),

                        "Add selection to": ("BOOLEAN", {"default": True, "label_on": "Positive", "label_off": "Negative"}),
                        "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
                        "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
                        "wildcard_mode": ("BOOLEAN", {"default": True, "label_on": "Populate", "label_off": "Fixed"}),

                        "positive_populated_text": ("STRING", {"multiline": True, "dynamicPrompts": False, 'placeholder': 'Populated Positive Prompt (Will be generated automatically)'}),
                        "negative_populated_text": ("STRING", {"multiline": True, "dynamicPrompts": False, 'placeholder': 'Populated Negative Prompt (Will be generated automatically)'}),

                        "token_normalization": (["none", "mean", "length", "length+mean"],),
                        "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"], {'default': 'comfy++'}),

                        "stop_at_clip_layer": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1}),
            
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                "optional": {
                        "vae_opt": ("VAE",)
                    },
                }

    CATEGORY = "InspirePack/Prompt"

    RETURN_TYPES = ("BASIC_PIPE", "STRING")
    RETURN_NAMES = ("basic_pipe", "cache_key")
    FUNCTION = "doit"

    def doit(self, **kwargs):
        pos_populated = kwargs['positive_populated_text']
        neg_populated = kwargs['negative_populated_text']

        clip_encoder = BNK_EncoderWrapper(kwargs['token_normalization'], kwargs['weight_interpretation'])

        if 'ImpactWildcardEncode' not in nodes.NODE_CLASS_MAPPINGS:
            utils.try_install_custom_node('https://github.com/ltdrdata/ComfyUI-Impact-Pack',
                                          "To use 'Make Basic Pipe (Inspire)' node, 'Impact Pack' extension is required.")
            raise Exception(f"[ERROR] To use 'Make Basic Pipe (Inspire)', you need to install 'Impact Pack'")

        model, clip, vae, key = CheckpointLoaderSimpleShared().doit(ckpt_name=kwargs['ckpt_name'], key_opt=kwargs['ckpt_key_opt'])
        clip = nodes.CLIPSetLastLayer().set_last_layer(clip, kwargs['stop_at_clip_layer'])[0]
        model, clip, positive = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardEncode'].process_with_loras(wildcard_opt=pos_populated, model=model, clip=clip, clip_encoder=clip_encoder)
        model, clip, negative = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardEncode'].process_with_loras(wildcard_opt=neg_populated, model=model, clip=clip, clip_encoder=clip_encoder)

        if 'vae_opt' in kwargs:
            vae = kwargs['vae_opt']

        basic_pipe = model, clip, vae, positive, negative

        return (basic_pipe, key)


class PromptBuilder:
    @classmethod
    def INPUT_TYPES(s):
        global prompt_builder_preset

        presets = ["#PRESET"]
        return {"required": {
                        "category": (list(prompt_builder_preset.keys()) + ["#PLACEHOLDER"], ),
                        "preset": (presets, ),
                        "text": ("STRING", {"multiline": True}),
                     },
                }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    def doit(self, **kwargs):
        return (kwargs['text'],)


class SeedExplorer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "seed_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": False}),
                "enable_additional": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "additional_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "additional_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_mode": (["GPU(=A1111)", "CPU"],),
                "initial_batch_seed_mode": (["incremental", "comfy"],),
            }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    @staticmethod
    def apply_variation(start_noise, seed_items, noise_device, mask=None):
        noise = start_noise
        for x in seed_items:
            if isinstance(x, str):
                item = x.split(':')
            else:
                item = x

            if len(item) == 2:
                try:
                    variation_seed = int(item[0])
                    variation_strength = float(item[1])

                    noise = utils.apply_variation_noise(noise, noise_device, variation_seed, variation_strength, mask=mask)
                except Exception:
                    print(f"[ERROR] IGNORED: SeedExplorer failed to processing '{x}'")
                    traceback.print_exc()
        return noise

    def doit(self, latent, seed_prompt, enable_additional, additional_seed, additional_strength, noise_mode,
             initial_batch_seed_mode):
        latent_image = latent["samples"]
        device = comfy.model_management.get_torch_device()
        noise_device = "cpu" if noise_mode == "CPU" else device

        seed_prompt = seed_prompt.replace("\n", "")
        items = seed_prompt.strip().split(",")

        if items == ['']:
            items = []

        if enable_additional:
            items.append((additional_seed, additional_strength))

        try:
            hd = items[0]
            tl = items[1:]

            if isinstance(hd, tuple):
                hd_seed = int(hd[0])
            else:
                hd_seed = int(hd)

            noise = utils.prepare_noise(latent_image, hd_seed, None, noise_device, initial_batch_seed_mode)
            noise = noise.to(device)
            noise = SeedExplorer.apply_variation(noise, tl, noise_device)
            noise = noise.cpu()

            return (noise,)

        except Exception:
            print(f"[ERROR] IGNORED: SeedExplorer failed")
            traceback.print_exc()

        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                            device=noise_device)
        return (noise,)


list_counter_map = {}


class ListCounter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "signal": (utils.any_typ,),
                    "base_value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("INT",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, signal, base_value, unique_id):
        if unique_id not in list_counter_map:
            count = 0
        else:
            count = list_counter_map[unique_id]

        list_counter_map[unique_id] = count + 1

        return (count + base_value, )


class CLIPTextEncodeWithWeight:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}), "clip": ("CLIP", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "add_weight": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                }
            }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "InspirePack/Util"

    def encode(self, clip, text, strength, add_weight):
        tokens = clip.tokenize(text)

        if add_weight != 0 or strength != 1:
            for v in tokens.values():
                for vv in v:
                    for i in range(0, len(vv)):
                        vv[i] = (vv[i][0], vv[i][1] * strength + add_weight)

        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )


class RandomGeneratorForList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "signal": (utils.any_typ,),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = (utils.any_typ, "INT",)
    RETURN_NAMES = ("signal", "random_value",)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, signal, seed, unique_id):
        if unique_id not in list_counter_map:
            count = 0
        else:
            count = list_counter_map[unique_id]

        list_counter_map[unique_id] = count + 1

        rn = random.Random()
        rn.seed(seed + count)
        new_seed = random.randint(0, 1125899906842624)

        return (signal, new_seed)


class RemoveControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, conditioning):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]

            if 'control' in n[1]:
                del n[1]['control']
            if 'control_apply_to_uncond' in n[1]:
                del n[1]['control_apply_to_uncond']
            c.append(n)

        return (c, )


class RemoveControlNetFromRegionalPrompts:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"regional_prompts": ("REGIONAL_PROMPTS", )}}
    RETURN_TYPES = ("REGIONAL_PROMPTS",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    def doit(self, regional_prompts):
        rcn = RemoveControlNet()
        res = []
        for rp in regional_prompts:
            _, _, _, _, positive, negative = rp.sampler.params
            positive, negative = rcn.doit(positive)[0], rcn.doit(negative)[0]
            sampler = rp.sampler.clone_with_conditionings(positive, negative)
            res.append(rp.clone_with_sampler(sampler))
        return (res, )


NODE_CLASS_MAPPINGS = {
    "LoadPromptsFromDir //Inspire": LoadPromptsFromDir,
    "LoadPromptsFromFile //Inspire": LoadPromptsFromFile,
    "LoadSinglePromptFromFile //Inspire": LoadSinglePromptFromFile,
    "UnzipPrompt //Inspire": UnzipPrompt,
    "ZipPrompt //Inspire": ZipPrompt,
    "PromptExtractor //Inspire": PromptExtractor,
    "GlobalSeed //Inspire": GlobalSeed,
    "GlobalSampler //Inspire": GlobalSampler,
    "BindImageListPromptList //Inspire": BindImageListPromptList,
    "WildcardEncode //Inspire": WildcardEncodeInspire,
    "PromptBuilder //Inspire": PromptBuilder,
    "SeedExplorer //Inspire": SeedExplorer,
    "ListCounter //Inspire": ListCounter,
    "CLIPTextEncodeWithWeight //Inspire": CLIPTextEncodeWithWeight,
    "RandomGeneratorForList //Inspire": RandomGeneratorForList,
    "MakeBasicPipe //Inspire": MakeBasicPipe,
    "RemoveControlNet //Inspire": RemoveControlNet,
    "RemoveControlNetFromRegionalPrompts //Inspire": RemoveControlNetFromRegionalPrompts,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPromptsFromDir //Inspire": "Load Prompts From Dir (Inspire)",
    "LoadPromptsFromFile //Inspire": "Load Prompts From File (Inspire)",
    "LoadSinglePromptFromFile //Inspire": "Load Single Prompt From File (Inspire)",
    "UnzipPrompt //Inspire": "Unzip Prompt (Inspire)",
    "ZipPrompt //Inspire": "Zip Prompt (Inspire)",
    "PromptExtractor //Inspire": "Prompt Extractor (Inspire)",
    "GlobalSeed //Inspire": "Global Seed (Inspire)",
    "GlobalSampler //Inspire": "Global Sampler (Inspire)",
    "BindImageListPromptList //Inspire": "Bind [ImageList, PromptList] (Inspire)",
    "WildcardEncode //Inspire": "Wildcard Encode (Inspire)",
    "PromptBuilder //Inspire": "Prompt Builder (Inspire)",
    "SeedExplorer //Inspire": "Seed Explorer (Inspire)",
    "ListCounter //Inspire": "List Counter (Inspire)",
    "CLIPTextEncodeWithWeight //Inspire": "CLIPTextEncodeWithWeight (Inspire)",
    "RandomGeneratorForList //Inspire": "Random Generator for List (Inspire)",
    "MakeBasicPipe //Inspire": "Make Basic Pipe (Inspire)",
    "RemoveControlNet //Inspire": "Remove ControlNet (Inspire)",
    "RemoveControlNetFromRegionalPrompts //Inspire": "Remove ControlNet [RegionalPrompts] (Inspire)"
}
