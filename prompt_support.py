import os
import re
import json
import sys

from PIL import Image
import nodes

import folder_paths
from server import PromptServer


class LoadPromptsFromDir:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            prompt_dir = os.path.join(current_directory, "prompts")
            prompt_dirs = [d for d in os.listdir(prompt_dir) if os.path.isdir(os.path.join(prompt_dir, d))]
        except Exception:
            prompt_dirs = []

        return {"required": {"prompt_dir": (prompt_dirs,)}}

    RETURN_TYPES = ("ZIPPED_PROMPT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/prompt"

    def doit(self, prompt_dir):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        prompt_dir = os.path.join(current_directory, "prompts", prompt_dir)
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
        try:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            prompt_dir = os.path.join(current_directory, "prompts")
            prompt_files = []
            for root, dirs, files in os.walk(prompt_dir):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, prompt_dir)
                        prompt_files.append(rel_path)
        except Exception:
            prompt_files = []

        return {"required": {"prompt_file": (prompt_files,)}}

    RETURN_TYPES = ("ZIPPED_PROMPT",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/prompt"

    def doit(self, prompt_file):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_directory, "prompts", prompt_file)

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


class UnzipPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"zipped_prompt": ("ZIPPED_PROMPT",), }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "name")

    FUNCTION = "doit"

    CATEGORY = "InspirePack/prompt"

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

    CATEGORY = "InspirePack/prompt"

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

    CATEGORY = "InspirePack/prompt"

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

        PromptServer.instance.send_sync("inspire-node-feedback", {"id": unique_id, "widget_name": "info", "type": "text", "data": text})
        return (positive, negative)


class GlobalSeed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": 1125899906842624}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "control_before_generate", "label_off": "control_after_generate"}),
                "action": (["fixed", "increment", "decrement", "randomize",
                            "increment for each node", "decrement for each node", "randomize for each node"], ),
                "last_seed": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "doit"

    CATEGORY = "InspirePack"

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

    CATEGORY = "InspirePack"

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


NODE_CLASS_MAPPINGS = {
    "LoadPromptsFromDir //Inspire": LoadPromptsFromDir,
    "LoadPromptsFromFile //Inspire": LoadPromptsFromFile,
    "UnzipPrompt //Inspire": UnzipPrompt,
    "ZipPrompt //Inspire": ZipPrompt,
    "PromptExtractor //Inspire": PromptExtractor,
    "GlobalSeed //Inspire": GlobalSeed,
    "BindImageListPromptList //Inspire": BindImageListPromptList,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPromptsFromDir //Inspire": "Load Prompts From Dir (Inspire)",
    "LoadPromptsFromFile //Inspire": "Load Prompts From File (Inspire)",
    "UnzipPrompt //Inspire": "Unzip Prompt (Inspire)",
    "ZipPrompt //Inspire": "Zip Prompt (Inspire)",
    "PromptExtractor //Inspire": "Prompt Extractor (Inspire)",
    "GlobalSeed //Inspire": "Global Seed (Inspire)",
    "BindImageListPromptList //Inspire": "Bind [ImageList, PromptList] (Inspire)",
}
