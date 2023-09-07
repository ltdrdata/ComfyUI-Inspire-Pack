import os
import re


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
                    prompt = file.read()

                    pattern = r"positive:(.*?)(?:\n*|$)negative:(.*)"
                    matches = re.search(pattern, prompt, re.DOTALL)

                    if matches:
                        positive_text = matches.group(1).strip()
                        negative_text = matches.group(2).strip()
                        result_tuple = (positive_text, negative_text, file_name)
                        prompts.append(result_tuple)
                    else:
                        print(f"[WARN] LoadPromptsFromFile: invalid prompt format in '{file_name}'")
            except Exception as e:
                print(f"[ERROR] LoadPromptsFromFile: an error occurred while processing '{file_name}': {str(e)}")

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


NODE_CLASS_MAPPINGS = {
    "LoadPromptsFromDir //Inspire": LoadPromptsFromDir,
    "UnzipPrompt //Inspire": UnzipPrompt,
    "ZipPrompt //Inspire": ZipPrompt,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPromptsFromDir //Inspire": "Load Prompts From Dir (Inspire)",
    "UnzipPrompt //Inspire": "Unzip Prompt (Inspire)",
    "ZipPrompt //Inspire": "Zip Prompt (Inspire)",
}
