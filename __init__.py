"""
@author: Dr.Lt.Data
@title: Inspire Pack
@nickname: Inspire Pack
@description: This extension provides various nodes to support Lora Block Weight and the Impact Pack.
"""

import importlib

print(f"### Loading: ComfyUI-Inspire-Pack (V0.27)")

node_list = [
    "lora_block_weight",
    "segs_support",
    "a1111_compat",
    "prompt_support",
    "inspire_server",
    "image_util",
    "regional_nodes",
    "sampler_nodes",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(".{}".format(module_name), __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
