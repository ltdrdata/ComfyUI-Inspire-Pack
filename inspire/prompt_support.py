import os
import re
import json
import shutil
import yaml

from PIL import Image
import nodes
import torch

import folder_paths
import comfy
import traceback
import random
import hashlib

from server import PromptServer
from .libs import utils, common
from .backend_support import CheckpointLoaderSimpleShared

import logging

model_path = folder_paths.models_dir
utils.add_folder_path_and_extensions("inspire_prompts", [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts"))], {'.txt'})


prompt_builder_preset = {}


resource_path = os.path.join(os.path.dirname(__file__), "..", "resources")
resource_path = os.path.abspath(resource_path)


try:
    pb_yaml_path = os.path.join(resource_path, 'prompt-builder.yaml')
    pb_yaml_path_example = os.path.join(resource_path, 'prompt-builder.yaml.example')

    if not os.path.exists(pb_yaml_path):
        shutil.copy(pb_yaml_path_example, pb_yaml_path)

    with open(pb_yaml_path, 'r', encoding="utf-8") as f:
        prompt_builder_preset = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:  # noqa: F841
    logging.error("[Inspire Pack] Failed to load 'prompt-builder.yaml'\nNOTE: Only files with UTF-8 encoding are supported.")


class LoadPromptsFromDir:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            prompt_dirs = []
            for x in folder_paths.get_folder_paths('inspire_prompts'):
                for d in os.listdir(x):
                    if os.path.isdir(os.path.join(x, d)):
                        prompt_dirs.append(d)
        except Exception:
            prompt_dirs = []

        return {"required": {
                    "prompt_dir": (prompt_dirs,)
                    },
                "optional": {
                    "reload": ("BOOLEAN", { "default": False, "label_on": "if file changed", "label_off": "if value changed"}),
                    "load_cap": ("INT", {"default": 0, "min": 0, "step": 1, "advanced": True, "tooltip": "The amount of prompts to load at once:\n0: Load all\n1 or higher: Load a specified number"}),
                    "start_index": ("INT", {"default": 0, "min": -1, "step": 1, "max": 0xffffffffffffffff, "advanced": True, "tooltip": "Starting index for loading prompts:\n-1: The last prompt\n0 or higher: Load from the specified index"}),
                    }
                }

    RETURN_TYPES = ("ZIPPED_PROMPT", "INT", "INT")
    RETURN_NAMES = ("zipped_prompt", "count", "remaining_count")
    OUTPUT_IS_LIST = (True, False, False)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    @staticmethod
    def IS_CHANGED(prompt_dir, reload=False, load_cap=0, start_index=-1):
        if not reload:
            return prompt_dir, load_cap, start_index
        else:
            candidates = []
            for d in folder_paths.get_folder_paths('inspire_prompts'):
                candidates.append(os.path.join(d, prompt_dir))

            prompt_files = []
            for x in candidates:
                for root, dirs, files in os.walk(x):
                    for file in files:
                        if file.endswith(".txt"):
                            prompt_files.append(os.path.join(root, file))

            prompt_files.sort()

            md5 = hashlib.md5()

            for file_name in prompt_files:
                md5.update(file_name.encode('utf-8'))
                with open(folder_paths.get_full_path('inspire_prompts', file_name), 'rb') as f:
                    while True:
                        chunk = f.read(4096)
                        if not chunk:
                            break
                        md5.update(chunk)

            return md5.hexdigest(), load_cap, start_index

    @staticmethod
    def doit(prompt_dir, reload=False, load_cap=0, start_index=-1):
        candidates = []
        for d in folder_paths.get_folder_paths('inspire_prompts'):
            candidates.append(os.path.join(d, prompt_dir))

        prompt_files = []
        for x in candidates:
            for root, dirs, files in os.walk(x):
                for file in files:
                    if file.endswith(".txt"):
                        prompt_files.append(os.path.join(root, file))

        prompt_files.sort()

        prompts = []
        for file_name in prompt_files:
            logging.info(f"file_name: {file_name}")
            try:
                with open(file_name, "r", encoding="utf-8") as file:
                    prompt_data = file.read()
                    prompt_list = re.split(r'\n\s*-+\s*\n', prompt_data)

                    for prompt in prompt_list:
                        pattern = r"^(?:(?:positive:(?P<positive>.*?)|negative:(?P<negative>.*?)|name:(?P<name>.*?))\n*)+$"
                        matches = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)

                        if matches:
                            positive_text = matches.group('positive').strip()
                            negative_text = matches.group('negative').strip()
                            name_text = matches.group('name').strip() if matches.group('name') else file_name
                            result_tuple = (positive_text, negative_text, name_text)
                            prompts.append(result_tuple)
                        else:
                            logging.warning(f"[Inspire Pack] LoadPromptsFromDir: invalid prompt format in '{file_name}'")
            except Exception as e:
                logging.error(f"[Inspire Pack] LoadPromptsFromDir: an error occurred while processing '{file_name}': {str(e)}\nNOTE: Only files with UTF-8 encoding are supported.")

        # slicing [start_index ~ start_index + load_cap]
        total_prompts = len(prompts)
        prompts = prompts[start_index:]
        remaining_count = False
        if load_cap > 0:
            remaining_count = max(0, len(prompts) - load_cap)
            prompts = prompts[:load_cap]

        return prompts, total_prompts, remaining_count


class LoadPromptsFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        prompt_files = []
        try:
            prompts_paths = folder_paths.get_folder_paths('inspire_prompts')
            for prompts_path in prompts_paths:
                for root, dirs, files in os.walk(prompts_path):
                    for file in files:
                        if file.endswith(".txt"):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, prompts_path)
                            prompt_files.append(rel_path)
        except Exception:
            prompt_files = []

        return {"required": {
                        "prompt_file": (prompt_files,)
                        },
                "optional": {
                        "text_data_opt": ("STRING", {"defaultInput": True}),
                        "reload": ("BOOLEAN", {"default": False, "label_on": "if file changed", "label_off": "if value changed"}),
                        "load_cap": ("INT", {"default": 0, "min": 0, "step": 1, "advanced": True, "tooltip": "The amount of prompts to load at once:\n0: Load all\n1 or higher: Load a specified number"}),
                        "start_index": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "step": 1, "advanced": True, "tooltip": "Starting index for loading prompts:\n-1: The last prompt\n0 or higher: Load from the specified index"}),
                        }
                }

    RETURN_TYPES = ("ZIPPED_PROMPT", "INT", "INT")
    RETURN_NAMES = ("zipped_prompt", "count", "remaining_count")
    OUTPUT_IS_LIST = (True, False, False)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    @staticmethod
    def IS_CHANGED(prompt_file, text_data_opt=None, reload=False, load_cap=0, start_index=-1):
        md5 = hashlib.md5()

        if text_data_opt is not None:
            md5.update(text_data_opt)
            return md5.hexdigest(), load_cap, start_index
        elif not reload:
            return prompt_file, load_cap, start_index
        else:
            matched_path = None
            for x in folder_paths.get_folder_paths('inspire_prompts'):
                matched_path = os.path.join(x, prompt_file)
                if not os.path.exists(matched_path):
                    matched_path = None
                else:
                    break

            if matched_path is None:
                return float('NaN')

            with open(matched_path, 'rb') as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    md5.update(chunk)

            return md5.hexdigest(), load_cap, start_index

    @staticmethod
    def doit(prompt_file, text_data_opt=None, reload=False, load_cap=0, start_index=-1):
        matched_path = None
        for d in folder_paths.get_folder_paths('inspire_prompts'):
            matched_path = os.path.join(d, prompt_file)
            if os.path.exists(matched_path):
                break
            else:
                matched_path = None

        if matched_path:
            logging.info(f"[Inspire Pack] LoadPromptsFromFile: file found '{prompt_file}'")
        else:
            logging.warning(f"[Inspire Pack] LoadPromptsFromFile: file not found '{prompt_file}'")

        prompts = []
        try:
            if not text_data_opt:
                with open(matched_path, "r", encoding="utf-8") as file:
                    prompt_data = file.read()
            else:
                prompt_data = text_data_opt

            prompt_list = re.split(r'\n\s*-+\s*\n', prompt_data)

            pattern = r"^(?:(?:positive:(?P<positive>.*?)|negative:(?P<negative>.*?)|name:(?P<name>.*?))\n*)+$"

            for p in prompt_list:
                matches = re.search(pattern, p, re.DOTALL)

                if matches:
                    positive_text = matches.group('positive').strip()
                    negative_text = matches.group('negative').strip()
                    name_text = matches.group('name').strip() if matches.group('name') else prompt_file
                    result_tuple = (positive_text, negative_text, name_text)
                    prompts.append(result_tuple)
                else:
                    logging.warning(f"[Inspire Pack] LoadPromptsFromFile: invalid prompt format in '{prompt_file}'")
        except Exception as e:
            logging.error(f"[Inspire Pack] LoadPromptsFromFile: an error occurred while processing '{prompt_file}': {str(e)}\nNOTE: Only files with UTF-8 encoding are supported.")

        # slicing [start_index ~ start_index + load_cap]
        total_prompts = len(prompts)
        prompts = prompts[start_index:]
        remaining_count = 0
        if load_cap > 0:
            remaining_count = max(0, len(prompts) - load_cap)
            prompts = prompts[:load_cap]

        return prompts, total_prompts, remaining_count


class LoadSinglePromptFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        prompt_files = []
        try:
            prompts_paths = folder_paths.get_folder_paths('inspire_prompts')
            for prompts_path in prompts_paths:
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
                    },
                "optional": {"text_data_opt": ("STRING", {"defaultInput": True})}
                }

    RETURN_TYPES = ("ZIPPED_PROMPT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    @staticmethod
    def doit(prompt_file, index, text_data_opt=None):
        prompt_path = None
        prompts_paths = folder_paths.get_folder_paths('inspire_prompts')
        for d in prompts_paths:
            prompt_path = os.path.join(d, prompt_file)
            if os.path.exists(prompt_path):
                break
            else:
                prompt_path = None

        if prompt_path:
            logging.info(f"[Inspire Pack] LoadSinglePromptFromFile: file found '{prompt_file}'")
        else:
            logging.warning(f"[Inspire Pack] LoadSinglePromptFromFile: file not found '{prompt_file}'")

        prompts = []
        try:
            if not text_data_opt:
                with open(prompt_path, "r", encoding="utf-8") as file:
                    prompt_data = file.read()
            else:
                prompt_data = text_data_opt

            prompt_list = re.split(r'\n\s*-+\s*\n', prompt_data)
            try:
                prompt = prompt_list[index]
            except Exception:
                prompt = prompt_list[-1]

            pattern = r"^(?:(?:positive:(?P<positive>.*?)|negative:(?P<negative>.*?)|name:(?P<name>.*?))\n*)+$"
            matches = re.search(pattern, prompt, re.DOTALL)

            if matches:
                positive_text = matches.group('positive').strip()
                negative_text = matches.group('negative').strip()
                name_text = matches.group('name').strip() if matches.group('name') else prompt_file
                result_tuple = (positive_text, negative_text, name_text)
                prompts.append(result_tuple)
            else:
                logging.warning(f"[Inspire Pack] LoadSinglePromptFromFile: invalid prompt format in '{prompt_file}'")
        except Exception as e:
            logging.error(f"[Inspire Pack] LoadSinglePromptFromFile: an error occurred while processing '{prompt_file}': {str(e)}\nNOTE: Only files with UTF-8 encoding are supported.")

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


prompt_blacklist = set(['filename_prefix'])


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


class SeedLogger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seeds": ("STRING", {"multiline": True, "dynamicPrompts": False, "control_after_generate": False}),
                "limit": ("INT", {"default": 5, "min": 0, "max": 0xffffffffffffffff}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ()
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    OUTPUT_NODE = True

    def doit(self, seed, seeds: str, limit, unique_id):

        if limit > 0:
            lines = seeds.split('\n')
            res = str(seed) + '\n' + '\n'.join(lines[:limit-1])
        else:
            res = str(seed) + '\n' + seeds

        PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": unique_id, "widget_name": "seeds", "type": "text", "data": res})
        return {}


class GlobalSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (common.SCHEDULERS, ),
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
            raise Exception("[ERROR] To use WildcardEncodeInspire, you need to install 'Advanced CLIP Text Encode'")
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

                        "mode": (["populate", "fixed", "reproduce"], {"default": "populate", "tooltip":
                            "populate: Before running the workflow, it overwrites the existing value of 'populated_text' with the prompt processed from 'wildcard_text'. In this mode, 'populated_text' cannot be edited.\n"
                            "fixed: Ignores wildcard_text and keeps 'populated_text' as is. You can edit 'populated_text' in this mode.\n"
                            "reproduce: This mode operates as 'fixed' mode only once for reproduction, and then it switches to 'populate' mode."
                                                                               }),

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
            raise Exception("[ERROR] To use 'Wildcard Encode (Inspire)', you need to install 'Impact Pack'")

        processed = []
        model, clip, conditioning = nodes.NODE_CLASS_MAPPINGS['ImpactWildcardEncode'].process_with_loras(wildcard_opt=populated, model=kwargs['model'], clip=kwargs['clip'], seed=kwargs['seed'], clip_encoder=clip_encoder, processed=processed)
        return (model, clip, conditioning, processed[0])


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
                        "wildcard_mode": (["populate", "fixed", "reproduce"], {"default": "populate", "tooltip":
                            "populate: Before running the workflow, it overwrites the existing value of 'populated_text' with the prompt processed from 'wildcard_text'. In this mode, 'populated_text' cannot be edited.\n"
                            "fixed: Ignores wildcard_text and keeps 'populated_text' as is. You can edit 'populated_text' in this mode.\n"
                            "reproduce: This mode operates as 'fixed' mode only once for reproduction, and then it switches to 'populate' mode."
                                                                               }),

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
            raise Exception("[ERROR] To use 'Make Basic Pipe (Inspire)', you need to install 'Impact Pack'")

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
            },
            "optional":
                {
                    "variation_method": (["linear", "slerp"],),
                    "model": ("MODEL",),
                }
        }

    RETURN_TYPES = ("NOISE_IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    @staticmethod
    def apply_variation(start_noise, seed_items, noise_device, mask=None, variation_method='linear'):
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

                    noise = utils.apply_variation_noise(noise, noise_device, variation_seed, variation_strength, mask=mask, variation_method=variation_method)
                except Exception:
                    logging.error(f"[Inspire Pack] IGNORED: SeedExplorer failed to processing '{x}'")
                    traceback.print_exc()
        return noise

    @staticmethod
    def doit(latent, seed_prompt, enable_additional, additional_seed, additional_strength, noise_mode,
             initial_batch_seed_mode, variation_method='linear', model=None):
        latent_image = latent["samples"]

        if hasattr(comfy.sample, 'fix_empty_latent_channels') and model is not None:
            latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

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
            noise = SeedExplorer.apply_variation(noise, tl, noise_device, variation_method=variation_method)
            noise = noise.cpu()

            return (noise,)

        except Exception:
            logging.error("[Inspire Pack] IGNORED: SeedExplorer failed")
            traceback.print_exc()

        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                            device=noise_device)
        return (noise,)


class CompositeNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("NOISE_IMAGE",),
                "source": ("NOISE_IMAGE",),
                "mode": (["center", "left-top", "right-top", "left-bottom", "right-bottom", "xy"], ),
                "x": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "y": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            },
        }

    RETURN_TYPES = ("NOISE_IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    def doit(self, destination, source, mode, x, y):
        new_tensor = destination.clone()

        if mode == 'center':
            y1 = (new_tensor.size(2) - source.size(2)) // 2
            x1 = (new_tensor.size(3) - source.size(3)) // 2
        elif mode == 'left-top':
            y1 = 0
            x1 = 0
        elif mode == 'right-top':
            y1 = 0
            x1 = new_tensor.size(2) - source.size(2)
        elif mode == 'left-bottom':
            y1 = new_tensor.size(3) - source.size(3)
            x1 = 0
        elif mode == 'right-bottom':
            y1 = new_tensor.size(3) - source.size(3)
            x1 = new_tensor.size(2) - source.size(2)
        else:  # mode == 'xy':
            x1 = max(0, x)
            y1 = max(0, y)

        # raw coordinates
        y2 = y1 + source.size(2)
        x2 = x1 + source.size(3)

        # bounding for destination
        top = max(0, y1)
        left = max(0, x1)
        bottom = min(new_tensor.size(2), y2)
        right = min(new_tensor.size(3), x2)

        # bounding for source
        left_gap = left - x1
        top_gap = top - y1

        width = right - left
        height = bottom - top

        height = min(height, y1 + source.size(2) - top)
        width = min(width, x1 + source.size(3) - left)

        # composite
        new_tensor[:, :, top:top + height, left:left + width] = source[:, :, top_gap:top_gap + height, left_gap:left_gap + width]

        return (new_tensor,)


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
    "SeedLogger //Inspire": SeedLogger,
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
    "CompositeNoise //Inspire": CompositeNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPromptsFromDir //Inspire": "Load Prompts From Dir (Inspire)",
    "LoadPromptsFromFile //Inspire": "Load Prompts From File (Inspire)",
    "LoadSinglePromptFromFile //Inspire": "Load Single Prompt From File (Inspire)",
    "UnzipPrompt //Inspire": "Unzip Prompt (Inspire)",
    "ZipPrompt //Inspire": "Zip Prompt (Inspire)",
    "PromptExtractor //Inspire": "Prompt Extractor (Inspire)",
    "GlobalSeed //Inspire": "Global Seed (Inspire)",
    "SeedLogger //Inspire": "Seed Logger (Inspire)",
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
    "RemoveControlNetFromRegionalPrompts //Inspire": "Remove ControlNet [RegionalPrompts] (Inspire)",
    "CompositeNoise //Inspire": "Composite Noise (Inspire)"
}
