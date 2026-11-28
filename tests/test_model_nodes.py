import os
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

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
    assert (icache_key, ccache_key) == ("insightface-buffalo_l-CPU", CLIP_VISION)
    assert pack.backend_support.cache[CLIP_VISION] == ("clipvision", (False, clip_vision))
    assert labels(pack) == [("7", 1, "IPADAPTER"), ("7", 2, "CLIP_VISION"), ("7", 4, "MODEL"), ("7", 5, "CLIP"), ("7", 3, "INSIGHTFACE")]


@pytest.mark.parametrize("cache_mode", ["insightface only", "clip_vision only", "all", "none"])
def test_the_preset_helper_loads_the_picked_insightface_model_in_every_cache_mode(pack, models, loaded, cache_mode):
    _, _, _, insightface, _, _, _, _ = pack.model_nodes.IPAdapterModelHelper().doit(
        "MODEL", "SD1.5 FaceID", 1.0, 1.0, "CUDA", clip=SimpleNamespace(tokenizer=None), cache_mode=cache_mode, unique_id="7", insightface_model_name="antelopev2")
    assert loaded[-1] == ("insightface", "CUDA", "antelopev2")
    assert insightface == ("INSIGHTFACE", "CUDA", "antelopev2")


@pytest.mark.parametrize("cache_mode", ["insightface only", "all"])
def test_the_cache_keeps_each_insightface_model_apart(pack, models, loaded, cache_mode):
    helper = pack.model_nodes.IPAdapterModelHelper()
    runs = [helper.doit("MODEL", "SD1.5 FaceID", 1.0, 1.0, "CPU", clip=SimpleNamespace(tokenizer=None), cache_mode=cache_mode, unique_id="7",
                        insightface_model_name=name) for name in ("buffalo_l", "antelopev2", "buffalo_l")]
    assert [run[3] for run in runs] == [("INSIGHTFACE", "CPU", "buffalo_l"), ("INSIGHTFACE", "CPU", "antelopev2"), ("INSIGHTFACE", "CPU", "buffalo_l")]
    assert [run[6] for run in runs] == ["insightface-buffalo_l-CPU", "insightface-antelopev2-CPU", "insightface-buffalo_l-CPU"]
    assert [call for call in loaded if call[0] == "insightface"] == [("insightface", "CPU", "buffalo_l"), ("insightface", "CPU", "antelopev2")]


@pytest.mark.parametrize("preset,clip,ipadapter", [("SD1.5 Plus Composition", None, "ip_plus_composition_sd15.safetensors"),
                                                   ("SDXL Plus Composition", "sdxl", "ip_plus_composition_sdxl.safetensors")])
def test_the_composition_presets_load_the_composition_adapters_with_vit_h(pack, models, loaded, tmp_path, preset, clip, ipadapter):
    for path in ("ipadapter/ip-adapter_sd15.safetensors", "ipadapter/ip-adapter_sdxl.safetensors", "ipadapter/ip_plus_composition_sd15.safetensors",
                 "ipadapter/ip_plus_composition_sdxl.safetensors", "clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"):
        (tmp_path / "models" / path).write_bytes(b"")
    pack.model_nodes.IPAdapterModelHelper().doit("MODEL", preset, 1.0, 1.0, "CPU", clip=sdxl_clip() if clip else None, unique_id="7")
    assert loaded == [("ipadapter", ipadapter), ("clip_vision", CLIP_VISION)]


@pytest.mark.parametrize("preset,clip,files,label", [("SD1.5 Plus Composition", None, [], (1, "IPADAPTER (FAIL)")),
                                                    ("SDXL", "sdxl", ["ipadapter/ip-adapter_sdxl.safetensors"], (2, "CLIP_VISION (FAIL)"))])
def test_a_preset_whose_ipadapter_or_clip_vision_file_is_missing_stops_before_loading(pack, models, loaded, tmp_path, preset, clip, files, label):
    for path in files:
        (tmp_path / "models" / path).write_bytes(b"")
    with pytest.raises(Exception, match="Failed to load several models"):
        pack.model_nodes.IPAdapterModelHelper().doit("MODEL", preset, 1.0, 1.0, "CPU", clip=sdxl_clip() if clip else None, unique_id="7")
    assert loaded == []
    assert ("7", *label) in labels(pack)


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
    assert (icache_key, ccache_key) == ("insightface-antelopev2-CUDA", "unit-clip-vision.safetensors")
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


PULID = "ip-adapter_pulid_sdxl_fp16.safetensors"


def pulid_folder(pack, tmp_path, monkeypatch, files):
    for name, key in files.items():
        (tmp_path / "pulid").mkdir(exist_ok=True)
        safetensors.torch.save_file({key: torch.zeros(1)}, str(tmp_path / "pulid" / name))
    monkeypatch.setattr(pack.folder_paths, "filename_list_cache", {})
    monkeypatch.setitem(pack.folder_paths.folder_names_and_paths, "pulid", ([str(tmp_path / "pulid")], pack.folder_paths.supported_pt_extensions))


@pytest.fixture
def pulid_loaders(pack, tmp_path, monkeypatch):
    import nodes
    pulid_folder(pack, tmp_path, monkeypatch, {PULID: "image_proj.mapping_0.0.weight"})
    calls = []

    class PulidModelLoader:
        def load_model(self, pulid_file):
            calls.append(("pulid", pulid_file))
            return (("PULID", pulid_file),)

    class PulidEvaClipLoader:
        def load_eva_clip(self):
            calls.append(("eva_clip",))
            return (("EVA_CLIP",),)

    class PulidInsightFaceLoader:
        def load_insightface(self, provider):
            calls.append(("face_analysis", provider))
            return (("FACEANALYSIS", provider),)

    for loader in (PulidModelLoader, PulidEvaClipLoader, PulidInsightFaceLoader):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, loader.__name__, loader)
    return calls


@pytest.mark.parametrize("cache_mode,second_run", [("insightface only", ["pulid", "eva_clip"]), ("eva_clip only", ["pulid", "face_analysis"]),
                                                   ("all", ["pulid"]), ("none", ["pulid", "eva_clip", "face_analysis"])])
def test_the_pulid_helper_loads_the_models_of_pulid_and_keeps_those_its_cache_mode_names(pack, pulid_loaders, cache_mode, second_run):
    helper = pack.model_nodes.PulidModelHelper()
    models = helper.doit(PULID, "CUDA", cache_mode)
    assert models == (("PULID", PULID), ("EVA_CLIP",), ("FACEANALYSIS", "CUDA"))
    assert pulid_loaders == [("pulid", PULID), ("eva_clip",), ("face_analysis", "CUDA")]
    pulid_loaders.clear()
    assert helper.doit(PULID, "CUDA", cache_mode) == models
    assert [call[0] for call in pulid_loaders] == second_run


def test_the_pulid_helper_caches_the_face_analysis_under_its_model_and_provider(pack, pulid_loaders):
    pack.model_nodes.PulidModelHelper().doit(PULID, "ROCM", "all")
    assert pack.backend_support.cache["insightface-antelopev2-ROCM"] == ("insightface", (False, ("FACEANALYSIS", "ROCM")))
    assert pack.backend_support.cache["pulid-eva_clip"] == ("eva_clip", (False, ("EVA_CLIP",)))


def test_the_pulid_helper_says_which_file_is_not_a_pulid_model_for_sdxl_before_it_loads_anything(pack, pulid_loaders, tmp_path, monkeypatch):
    pulid_folder(pack, tmp_path, monkeypatch, {"pulid_flux_v0.9.1.safetensors": "pulid_encoder.id_embedding_mapping.0.weight"})
    with pytest.raises(Exception, match="pulid_flux_v0.9.1.safetensors is not a PuLID model for SDXL"):
        pack.model_nodes.PulidModelHelper().doit("pulid_flux_v0.9.1.safetensors", "CPU", "none")
    assert pulid_loaders == []


def test_the_pulid_helper_asks_for_pulid_comfyui_when_it_is_missing(pack, monkeypatch):
    requests = []
    monkeypatch.setattr(pack.utils, "try_install_custom_node", lambda url, msg: requests.append(url))
    with pytest.raises(Exception, match="install 'PuLID_ComfyUI'"):
        pack.model_nodes.PulidModelHelper().doit(PULID, "CPU", "none")
    assert requests == ["https://github.com/cubiq/PuLID_ComfyUI"]


@pytest.mark.parametrize("folder", ["pulid", "instantid"])
def test_the_model_folder_of_an_adapter_family_is_listed_without_its_pack(pack, folder):
    folder_paths = pack.folder_paths
    assert os.path.join(folder_paths.models_dir, folder) in folder_paths.get_folder_paths(folder)
    assert ".safetensors" in folder_paths.folder_names_and_paths[folder][1]


INSTANTID_CN = "instantid/diffusion_pytorch_model.safetensors"


@pytest.fixture
def instantid_loaders(pack, monkeypatch):
    import nodes
    calls = []

    class InstantIDModelLoader:
        def load_model(self, instantid_file):
            calls.append(("instantid", instantid_file))
            return (("INSTANTID", instantid_file),)

    class InstantIDFaceAnalysis:
        def load_insight_face(self, provider):
            calls.append(("insightface", provider))
            return (("FACEANALYSIS", provider),)

    class ControlNetLoader:
        def load_controlnet(self, control_net_name):
            calls.append(("control_net", control_net_name))
            return (("CONTROL_NET", control_net_name),)

    for loader in (InstantIDModelLoader, InstantIDFaceAnalysis, ControlNetLoader):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, loader.__name__, loader)
    return calls


@pytest.mark.parametrize("cache_mode,second_run", [("insightface only", ["instantid", "control_net"]), ("none", ["instantid", "insightface", "control_net"])])
def test_the_instantid_helper_loads_the_models_of_instantid_and_keeps_those_its_cache_mode_names(pack, instantid_loaders, cache_mode, second_run):
    helper = pack.model_nodes.InstantIDModelHelper()
    models = helper.doit("ip-adapter.bin", INSTANTID_CN, "CoreML", cache_mode)
    assert models == (("INSTANTID", "ip-adapter.bin"), ("FACEANALYSIS", "CoreML"), ("CONTROL_NET", INSTANTID_CN))
    assert instantid_loaders == [("instantid", "ip-adapter.bin"), ("insightface", "CoreML"), ("control_net", INSTANTID_CN)]
    instantid_loaders.clear()
    assert helper.doit("ip-adapter.bin", INSTANTID_CN, "CoreML", cache_mode) == models
    assert [call[0] for call in instantid_loaders] == second_run


def test_the_pulid_and_instantid_helpers_share_the_cached_face_analysis(pack, pulid_loaders, instantid_loaders):
    _, _, face_analysis = pack.model_nodes.PulidModelHelper().doit(PULID, "CPU", "insightface only")
    _, insightface, _ = pack.model_nodes.InstantIDModelHelper().doit("ip-adapter.bin", INSTANTID_CN, "CPU", "insightface only")
    assert insightface is face_analysis
    assert [call[0] for call in instantid_loaders] == ["instantid", "control_net"]


def test_the_instantid_helper_asks_for_comfyui_instantid_when_it_is_missing(pack, monkeypatch):
    requests = []
    monkeypatch.setattr(pack.utils, "try_install_custom_node", lambda url, msg: requests.append(url))
    with pytest.raises(Exception, match="install 'ComfyUI InstantID'"):
        pack.model_nodes.InstantIDModelHelper().doit("ip-adapter.bin", INSTANTID_CN, "CPU", "none")
    assert requests == ["https://github.com/cubiq/ComfyUI_InstantID"]
