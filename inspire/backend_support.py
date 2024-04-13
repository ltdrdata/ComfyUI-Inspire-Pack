import json
import os

import folder_paths
import nodes
from server import PromptServer

from .libs.utils import TaggedCache, any_typ

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
settings_file = os.path.join(root_dir, 'cache_settings.json')
try:
    with open(settings_file) as f:
        cache_settings = json.load(f)
except Exception as e:
    print(e)
    cache_settings = {}
cache = TaggedCache(cache_settings)
cache_count = {}


def update_cache(k, tag, v):
    cache[k] = (tag, v)
    cnt = cache_count.get(k)
    if cnt is None:
        cnt = 0
        cache_count[k] = cnt
    else:
        cache_count[k] += 1


def cache_weak_hash(k):
    cnt = cache_count.get(k)
    if cnt is None:
        cnt = 0

    return k, cnt


class CacheBackendData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False, "placeholder": "Input data key (e.g. 'model a', 'chunli lora', 'girl latent 3', ...)"}),
                "tag": ("STRING", {"multiline": False, "placeholder": "Tag: short description"}),
                "data": (any_typ,),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("data opt",)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    OUTPUT_NODE = True

    def doit(self, key, tag, data):
        global cache

        if key == '*':
            print(f"[Inspire Pack] CacheBackendData: '*' is reserved key. Cannot use that key")

        update_cache(key, tag, (False, data))
        return (data,)


class CacheBackendDataNumberKey:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "tag": ("STRING", {"multiline": False, "placeholder": "Tag: short description"}),
                "data": (any_typ,),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("data opt",)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    OUTPUT_NODE = True

    def doit(self, key, tag, data):
        global cache

        update_cache(key, tag, (False, data))
        return (data,)


class CacheBackendDataList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False, "placeholder": "Input data key (e.g. 'model a', 'chunli lora', 'girl latent 3', ...)"}),
                "tag": ("STRING", {"multiline": False, "placeholder": "Tag: short description"}),
                "data": (any_typ,),
            }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("data opt",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    OUTPUT_NODE = True

    def doit(self, key, tag, data):
        global cache

        if key == '*':
            print(f"[Inspire Pack] CacheBackendDataList: '*' is reserved key. Cannot use that key")

        update_cache(key[0], tag[0], (True, data))
        return (data,)


class CacheBackendDataNumberKeyList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "tag": ("STRING", {"multiline": False, "placeholder": "Tag: short description"}),
                "data": (any_typ,),
            }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("data opt",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    OUTPUT_NODE = True

    def doit(self, key, tag, data):
        global cache
        update_cache(key[0], tag[0], (True, data))
        return (data,)


class RetrieveBackendData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False, "placeholder": "Input data key (e.g. 'model a', 'chunli lora', 'girl latent 3', ...)"}),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("data",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    @staticmethod
    def doit(key):
        global cache

        v = cache.get(key)

        if v is None:
            print(f"[RetrieveBackendData] '{key}' is unregistered key.")
            return (None,)

        is_list, data = v[1]

        if is_list:
            return (data,)
        else:
            return ([data],)

    @staticmethod
    def IS_CHANGED(key):
        return cache_weak_hash(key)


class RetrieveBackendDataNumberKey(RetrieveBackendData):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }


class RemoveBackendData:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"multiline": False, "placeholder": "Input data key ('*' = clear all)"}),
            },
            "optional": {
                "signal_opt": (any_typ,),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal",)

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    OUTPUT_NODE = True

    @staticmethod
    def doit(key, signal_opt=None):
        global cache

        if key == '*':
            cache = TaggedCache(cache_settings)
        elif key in cache:
            del cache[key]
        else:
            print(f"[Inspire Pack] RemoveBackendData: invalid data key {key}")

        return (signal_opt,)


class RemoveBackendDataNumberKey(RemoveBackendData):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "signal_opt": (any_typ,),
            }
        }

    @staticmethod
    def doit(key, signal_opt=None):
        global cache

        if key in cache:
            del cache[key]
        else:
            print(f"[Inspire Pack] RemoveBackendDataNumberKey: invalid data key {key}")

        return (signal_opt,)


class ShowCachedInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cache_info": ("STRING", {"multiline": True, "default": ""}),
                "key": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()

    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    OUTPUT_NODE = True

    @staticmethod
    def get_data():
        global cache

        text1 = "---- [String Key Caches] ----\n"
        text2 = "---- [Number Key Caches] ----\n"
        for k, v in cache.items():
            tag = 'N/A(tag)' if v[0] == '' else v[0]
            if isinstance(k, str):
                text1 += f'{k}: {tag}\n'
            else:
                text2 += f'{k}: {tag}\n'

        text3 = "---- [TagCache Settings] ----\n"
        for k, v in cache._tag_settings.items():
            text3 += f'{k}: {v}\n'

        for k, v in cache._data.items():
            if k not in cache._tag_settings:
                text3 += f'{k}: {v.maxsize}\n'

        return f'{text1}\n{text2}\n{text3}'

    @staticmethod
    def set_cache_settings(data: str):
        global cache
        settings = data.split("---- [TagCache Settings] ----\n")[-1].strip().split("\n")

        new_tag_settings = {}
        for s in settings:
            k, v = s.split(":")
            new_tag_settings[k] = int(v.strip())
        if new_tag_settings == cache._tag_settings:
            # tag settings is not changed
            return

        # print(f'set to {new_tag_settings}')
        new_cache = TaggedCache(new_tag_settings)
        for k, v in cache.items():
            new_cache[k] = v
        cache = new_cache

    def doit(self, cache_info, key, unique_id):
        text = ShowCachedInfo.get_data()
        PromptServer.instance.send_sync("inspire-node-feedback", {"node_id": unique_id, "widget_name": "cache_info", "type": "text", "data": text})

        return {}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


class CheckpointLoaderSimpleShared(nodes.CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                    "key_opt": ("STRING", {"multiline": False, "placeholder": "If empty, use 'ckpt_name' as the key."}),
                },
                "optional": {
                    "mode": (['Auto', 'Override Cache', 'Read Only'],),
                }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "cache key")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    def doit(self, ckpt_name, key_opt, mode='Auto'):
        if mode == 'Read Only':
            if key_opt.strip() == '':
                raise Exception("[CheckpointLoaderSimpleShared] key_opt cannot be omit if mode is 'Read Only'")
            key = key_opt.strip()
        elif key_opt.strip() == '':
            key = ckpt_name
        else:
            key = key_opt.strip()

        if key not in cache or mode == 'Override Cache':
            res = self.load_checkpoint(ckpt_name)
            update_cache(key, "ckpt", (False, res))
            cache_kind = 'ckpt'
            print(f"[Inspire Pack] CheckpointLoaderSimpleShared: Ckpt '{ckpt_name}' is cached to '{key}'.")
        else:
            cache_kind, (_, res) = cache[key]
            print(f"[Inspire Pack] CheckpointLoaderSimpleShared: Cached ckpt '{key}' is loaded. (Loading skip)")

        if cache_kind == 'ckpt':
            model, clip, vae = res
        elif cache_kind == 'unclip_ckpt':
            model, clip, vae, _ = res
        else:
            raise Exception(f"[CheckpointLoaderSimpleShared] Unexpected cache_kind '{cache_kind}'")

        return model, clip, vae, key

    @staticmethod
    def IS_CHANGED(ckpt_name, key_opt, mode='Auto'):
        if mode == 'Read Only':
            if key_opt.strip() == '':
                raise Exception("[CheckpointLoaderSimpleShared] key_opt cannot be omit if mode is 'Read Only'")
            key = key_opt.strip()
        elif key_opt.strip() == '':
            key = ckpt_name
        else:
            key = key_opt.strip()

        if mode == 'Read Only':
            return (None, cache_weak_hash(key))
        elif mode == 'Override Cache':
            return (ckpt_name, key)

        return (None, cache_weak_hash(key))


class StableCascade_CheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        ckpts = folder_paths.get_filename_list("checkpoints")
        default_stage_b = ''
        default_stage_c = ''

        sc_ckpts = [x for x in ckpts if 'cascade' in x.lower()]
        sc_b_ckpts = [x for x in sc_ckpts if 'stage_b' in x.lower()]
        sc_c_ckpts = [x for x in sc_ckpts if 'stage_c' in x.lower()]

        if len(sc_b_ckpts) == 0:
            sc_b_ckpts = [x for x in ckpts if 'stage_b' in x.lower()]
        if len(sc_c_ckpts) == 0:
            sc_c_ckpts = [x for x in ckpts if 'stage_c' in x.lower()]

        if len(sc_b_ckpts) == 0:
            sc_b_ckpts = ckpts
        if len(sc_c_ckpts) == 0:
            sc_c_ckpts = ckpts

        if len(sc_b_ckpts) > 0:
            default_stage_b = sc_b_ckpts[0]
        if len(sc_c_ckpts) > 0:
            default_stage_c = sc_c_ckpts[0]

        return {"required": {
                        "stage_b": (ckpts, {'default': default_stage_b}),
                        "key_opt_b": ("STRING", {"multiline": False, "placeholder": "If empty, use 'stage_b' as the key."}),
                        "stage_c": (ckpts, {'default': default_stage_c}),
                        "key_opt_c": ("STRING", {"multiline": False, "placeholder": "If empty, use 'stage_c' as the key."}),
                        "cache_mode": (["none", "stage_b", "stage_c", "all"], {"default": "none"}),
                     }}

    RETURN_TYPES = ("MODEL", "VAE", "MODEL", "VAE", "CLIP_VISION", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("b_model", "b_vae", "c_model", "c_vae", "c_clip_vision", "clip", "key_b", "key_c")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    def doit(self, stage_b, key_opt_b, stage_c, key_opt_c, cache_mode):
        if key_opt_b.strip() == '':
            key_b = stage_b
        else:
            key_b = key_opt_b.strip()

        if key_opt_c.strip() == '':
            key_c = stage_c
        else:
            key_c = key_opt_c.strip()

        if cache_mode in ['stage_b', "all"]:
            if key_b not in cache:
                res_b = nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name=stage_b)
                update_cache(key_b, "ckpt", (False, res_b))
                print(f"[Inspire Pack] StableCascade_CheckpointLoader: Ckpt '{stage_b}' is cached to '{key_b}'.")
            else:
                _, (_, res_b) = cache[key_b]
                print(f"[Inspire Pack] StableCascade_CheckpointLoader: Cached ckpt '{key_b}' is loaded. (Loading skip)")
            b_model, clip, b_vae = res_b
        else:
            b_model, clip, b_vae = nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name=stage_b)

        if cache_mode in ['stage_c', "all"]:
            if key_c not in cache:
                res_c = nodes.unCLIPCheckpointLoader().load_checkpoint(ckpt_name=stage_c)
                update_cache(key_c, "unclip_ckpt", (False, res_c))
                print(f"[Inspire Pack] StableCascade_CheckpointLoader: Ckpt '{stage_c}' is cached to '{key_c}'.")
            else:
                _, (_, res_c) = cache[key_c]
                print(f"[Inspire Pack] StableCascade_CheckpointLoader: Cached ckpt '{key_c}' is loaded. (Loading skip)")
            c_model, _, c_vae, clip_vision = res_c
        else:
            c_model, _, c_vae, clip_vision = nodes.unCLIPCheckpointLoader().load_checkpoint(ckpt_name=stage_c)

        return b_model, b_vae, c_model, c_vae, clip_vision, clip, key_b, key_c


NODE_CLASS_MAPPINGS = {
    "CacheBackendData //Inspire": CacheBackendData,
    "CacheBackendDataNumberKey //Inspire": CacheBackendDataNumberKey,
    "CacheBackendDataList //Inspire": CacheBackendDataList,
    "CacheBackendDataNumberKeyList //Inspire": CacheBackendDataNumberKeyList,
    "RetrieveBackendData //Inspire": RetrieveBackendData,
    "RetrieveBackendDataNumberKey //Inspire": RetrieveBackendDataNumberKey,
    "RemoveBackendData //Inspire": RemoveBackendData,
    "RemoveBackendDataNumberKey //Inspire": RemoveBackendDataNumberKey,
    "ShowCachedInfo //Inspire": ShowCachedInfo,
    "CheckpointLoaderSimpleShared //Inspire": CheckpointLoaderSimpleShared,
    "StableCascade_CheckpointLoader //Inspire": StableCascade_CheckpointLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CacheBackendData //Inspire": "Cache Backend Data (Inspire)",
    "CacheBackendDataNumberKey //Inspire": "Cache Backend Data [NumberKey] (Inspire)",
    "CacheBackendDataList //Inspire": "Cache Backend Data List (Inspire)",
    "CacheBackendDataNumberKeyList //Inspire": "Cache Backend Data List [NumberKey] (Inspire)",
    "RetrieveBackendData //Inspire": "Retrieve Backend Data (Inspire)",
    "RetrieveBackendDataNumberKey //Inspire": "Retrieve Backend Data [NumberKey] (Inspire)",
    "RemoveBackendData //Inspire": "Remove Backend Data (Inspire)",
    "RemoveBackendDataNumberKey //Inspire": "Remove Backend Data [NumberKey] (Inspire)",
    "ShowCachedInfo //Inspire": "Show Cached Info (Inspire)",
    "CheckpointLoaderSimpleShared //Inspire": "Shared Checkpoint Loader (Inspire)",
    "StableCascade_CheckpointLoader //Inspire": "Stable Cascade Checkpoint Loader (Inspire)"
}
