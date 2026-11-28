from types import SimpleNamespace

import pytest

CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
EXTRA_MODEL_PATHS = """unit:
  base_path: {base}
  ipadapter: ipadapter/
  clip_vision: clip_vision/
  loras: loras/
"""


@pytest.fixture
def models(pack, tmp_path, monkeypatch):
    for path in ("models/ipadapter/ip-adapter-faceid_sd15.bin", "models/ipadapter/notes.txt", "models/clip_vision/" + CLIP_VISION,
                 "models/loras/ip-adapter-faceid_sd15_lora.safetensors", "extra/ipadapter/unit/unit-ipadapter.safetensors",
                 "extra/clip_vision/unit-clip-vision.safetensors", "extra/loras/unit-lora.safetensors"):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(b"")
    folder_paths = pack.folder_paths
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    for name in ("ipadapter", "clip_vision", "loras"):
        monkeypatch.setitem(folder_paths.folder_names_and_paths, name, ([str(tmp_path / "models" / name)], folder_paths.supported_pt_extensions))
    import utils.extra_config
    (tmp_path / "extra_model_paths.yaml").write_text(EXTRA_MODEL_PATHS.format(base=tmp_path / "extra"))
    utils.extra_config.load_extra_path_config(str(tmp_path / "extra_model_paths.yaml"))


@pytest.fixture
def loaded(pack, monkeypatch):
    import nodes
    calls = []

    class IPAdapterModelLoader:
        def load_ipadapter_model(self, ipadapter_file):
            calls.append(("ipadapter", ipadapter_file))
            return (("IPADAPTER", ipadapter_file),)

    class IPAdapterInsightFaceLoader:
        def load_insightface(self, provider, model_name):
            calls.append(("insightface", provider, model_name))
            return (("INSIGHTFACE", provider, model_name),)

    class CLIPVisionLoader:
        def load_clip(self, clip_name):
            calls.append(("clip_vision", clip_name))
            return (("CLIP_VISION", clip_name),)

    class LoraLoader:
        def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
            calls.append(("lora", model, clip, lora_name, strength_model, strength_clip))
            return (model, lora_name), (clip, lora_name)

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "IPAdapter", object)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "IPAdapterModelLoader", IPAdapterModelLoader)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "IPAdapterInsightFaceLoader", IPAdapterInsightFaceLoader)
    monkeypatch.setattr(nodes, "CLIPVisionLoader", CLIPVisionLoader)
    monkeypatch.setattr(nodes, "LoraLoader", LoraLoader)
    return calls


def labels(pack):
    return [(data["node_id"], data["output_idx"], data["label"]) for event, data in pack.prompt_server.sent if event == "inspire-node-output-label"]


def sdxl_clip():
    from comfy import sdxl_clip
    return SimpleNamespace(tokenizer=object.__new__(sdxl_clip.SDXLTokenizer))


def test_the_preset_helper_loads_the_files_of_its_preset(pack, models, loaded):
    clip = SimpleNamespace(tokenizer=None)
    pipe, ipadapter, clip_vision, insightface, model, lora_clip, icache_key, ccache_key = pack.model_nodes.IPAdapterModelHelper().doit(
        "MODEL", "SD1.5 FaceID", 0.6, 0.4, "CPU", clip=clip, cache_mode="all", unique_id="7")
    assert loaded == [("ipadapter", "ip-adapter-faceid_sd15.bin"), ("clip_vision", CLIP_VISION),
                      ("lora", "MODEL", clip, "ip-adapter-faceid_sd15_lora.safetensors", 0.6, 0.4), ("insightface", "CPU", "buffalo_l")]
    assert (ipadapter, clip_vision, insightface) == (("IPADAPTER", "ip-adapter-faceid_sd15.bin"), ("CLIP_VISION", CLIP_VISION), ("INSIGHTFACE", "CPU", "buffalo_l"))
    assert (model, lora_clip) == (("MODEL", "ip-adapter-faceid_sd15_lora.safetensors"), (clip, "ip-adapter-faceid_sd15_lora.safetensors"))
    assert pipe[:4] == (ipadapter, model, clip_vision, insightface)
    assert (icache_key, ccache_key) == ("insightface-CPU", CLIP_VISION)
    assert pack.backend_support.cache[CLIP_VISION] == ("clipvision", (False, clip_vision))
    assert labels(pack) == [("7", 1, "IPADAPTER"), ("7", 2, "CLIP_VISION"), ("7", 4, "MODEL"), ("7", 5, "CLIP"), ("7", 3, "INSIGHTFACE")]


@pytest.mark.parametrize("cache_mode", ["insightface only", "clip_vision only", "all", "none"])
def test_the_preset_helper_loads_the_picked_insightface_model_in_every_cache_mode(pack, models, loaded, cache_mode):
    _, _, _, insightface, _, _, _, _ = pack.model_nodes.IPAdapterModelHelper().doit(
        "MODEL", "SD1.5 FaceID", 1.0, 1.0, "CUDA", clip=SimpleNamespace(tokenizer=None), cache_mode=cache_mode, unique_id="7", insightface_model_name="antelopev2")
    assert loaded[-1] == ("insightface", "CUDA", "antelopev2")
    assert insightface == ("INSIGHTFACE", "CUDA", "antelopev2")


@pytest.mark.parametrize("preset,clip,ipadapter", [("SD1.5 Plus Composition", None, "ip_plus_composition_sd15.safetensors"),
                                                   ("SDXL Plus Composition", "sdxl", "ip_plus_composition_sdxl.safetensors")])
def test_the_composition_presets_load_the_composition_adapters_with_vit_h(pack, models, loaded, tmp_path, preset, clip, ipadapter):
    for path in ("ipadapter/ip-adapter_sd15.safetensors", "ipadapter/ip-adapter_sdxl.safetensors", "ipadapter/ip_plus_composition_sd15.safetensors",
                 "ipadapter/ip_plus_composition_sdxl.safetensors", "clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"):
        (tmp_path / "models" / path).write_bytes(b"")
    pack.model_nodes.IPAdapterModelHelper().doit("MODEL", preset, 1.0, 1.0, "CPU", clip=sdxl_clip() if clip else None, unique_id="7")
    assert loaded == [("ipadapter", ipadapter), ("clip_vision", CLIP_VISION)]


@pytest.mark.parametrize("preset,clip", [("SDXL", None), ("SD1.5", "sdxl")])
def test_the_preset_helper_refuses_to_mix_sdxl_and_sd15(pack, models, loaded, preset, clip):
    with pytest.raises(Exception, match="cannot mix SDXL and SD1.5"):
        pack.model_nodes.IPAdapterModelHelper().doit("MODEL", preset, 1.0, 1.0, "CPU", clip=sdxl_clip() if clip else None, unique_id="7")
    assert loaded == []
    assert labels(pack) == [("7", 1, "IPADAPTER (fail)"), ("7", 2, "CLIP_VISION (fail)"), ("7", 3, "INSIGHTFACE (fail)"), ("7", 4, "MODEL (fail)"), ("7", 5, "CLIP (fail)")]


def test_the_manual_helper_lists_the_model_files_also_of_the_folders_of_extra_model_paths(pack, models):
    required = pack.model_nodes.IPAdapterModelHelperManual.INPUT_TYPES()["required"]
    assert required["ipadapter_model"][0] == ["ip-adapter-faceid_sd15.bin", "unit/unit-ipadapter.safetensors"]
    assert required["clip_vision_model"][0] == [CLIP_VISION, "unit-clip-vision.safetensors"]
    assert required["lora_model"][0] == ["None", "ip-adapter-faceid_sd15_lora.safetensors", "unit-lora.safetensors"]


def test_the_manual_helper_loads_the_picked_files(pack, models, loaded):
    pipe, ipadapter, clip_vision, insightface, model, clip, icache_key, ccache_key = pack.model_nodes.IPAdapterModelHelperManual().doit(
        "MODEL", "unit/unit-ipadapter.safetensors", "unit-clip-vision.safetensors", "unit-lora.safetensors", True, 0.6, 0.4, "CUDA",
        clip="CLIP", cache_mode="all", unique_id="9", insightface_model_name="antelopev2")
    assert loaded == [("ipadapter", "unit/unit-ipadapter.safetensors"), ("clip_vision", "unit-clip-vision.safetensors"),
                      ("lora", "MODEL", "CLIP", "unit-lora.safetensors", 0.6, 0.4), ("insightface", "CUDA", "antelopev2")]
    assert (ipadapter, clip_vision, insightface) == (("IPADAPTER", "unit/unit-ipadapter.safetensors"), ("CLIP_VISION", "unit-clip-vision.safetensors"), ("INSIGHTFACE", "CUDA", "antelopev2"))
    assert (model, clip) == (("MODEL", "unit-lora.safetensors"), ("CLIP", "unit-lora.safetensors"))
    assert pipe[:4] == (ipadapter, model, clip_vision, insightface)
    assert (icache_key, ccache_key) == ("insightface-CUDA", "unit-clip-vision.safetensors")
    assert labels(pack) == [("9", 3, "INSIGHTFACE")]


def test_the_manual_helper_without_a_lora_or_insightface_passes_the_model_on_whatever_the_checkpoint(pack, models, loaded):
    clip = sdxl_clip()
    pipe, _, _, insightface, model, clip_out, icache_key, ccache_key = pack.model_nodes.IPAdapterModelHelperManual().doit(
        "MODEL", "ip-adapter-faceid_sd15.bin", CLIP_VISION, "None", False, 1.0, 1.0, "CPU", clip=clip, cache_mode="insightface only", unique_id="9")
    assert [call[0] for call in loaded] == ["ipadapter", "clip_vision"]
    assert (insightface, model, clip_out, icache_key, ccache_key) == (None, "MODEL", clip, "", "")
    assert pipe[4]("other model") == "other model"
    assert labels(pack) == [("9", 3, "INSIGHTFACE (N/A)")]


def test_both_helpers_ask_for_ipadapter_plus_before_anything_else(pack, models, monkeypatch):
    import nodes
    monkeypatch.delitem(nodes.NODE_CLASS_MAPPINGS, "IPAdapter", raising=False)
    helpers = pack.model_nodes
    with pytest.raises(Exception, match="install 'ComfyUI IPAdapter Plus'"):
        helpers.IPAdapterModelHelper().doit("MODEL", "SDXL", 1.0, 1.0, "CPU", unique_id="7")
    with pytest.raises(Exception, match="install 'ComfyUI IPAdapter Plus'"):
        helpers.IPAdapterModelHelperManual().doit("MODEL", "ip-adapter-faceid_sd15.bin", CLIP_VISION, "None", False, 1.0, 1.0, "CPU", unique_id="9")
    assert labels(pack) == []
