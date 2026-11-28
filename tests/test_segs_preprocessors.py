import inspect

import pytest
import torch

AUX_URL = "https://github.com/Fannovel16/comfyui_controlnet_aux"
OUTPUT = torch.zeros(1, 8, 8, 3)

AUX_NODES = {
    "DSINE-NormalMapPreprocessor": lambda image, fov=60.0, iterations=5, resolution=512: None,
    "BAE-NormalMapPreprocessor": lambda image, resolution=512: None,
    "Metric3D-NormalMapPreprocessor": lambda image, backbone="vit-small", fx=1000, fy=1000, resolution=512: None,
    "DepthAnythingPreprocessor": lambda image, ckpt_name="depth_anything_vitl14.pth", resolution=512: None,
    "DepthAnythingV2Preprocessor": lambda image, ckpt_name="depth_anything_v2_vitl.pth", resolution=512: None,
}
V1_FILES = ["depth_anything_vitl14.pth", "depth_anything_vitb14.pth", "depth_anything_vits14.pth"]
V2_FILES = ["depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"]


def stand_in(name, signature, calls):
    class Preprocessor:
        def execute(self, *args, **kwargs):
            bound = inspect.signature(signature).bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            del arguments["image"]
            calls.append((name, arguments))
            return (OUTPUT,)
    return Preprocessor


@pytest.fixture
def aux(monkeypatch):
    import nodes
    calls = []
    for name, signature in AUX_NODES.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, stand_in(name, signature, calls))
    return calls


def picture():
    return torch.rand(1, 100, 150, 3)


@pytest.mark.parametrize("model_type,node,arguments", [
    ("DSINE", "DSINE-NormalMapPreprocessor", {"fov": 30.0, "iterations": 7}),
    ("BAE", "BAE-NormalMapPreprocessor", {}),
    ("Metric3D", "Metric3D-NormalMapPreprocessor", {"backbone": "vit-large", "fx": 800, "fy": 600}),
])
def test_normal_map_runs_the_node_of_its_model_with_the_widgets_of_that_model(pack, run, aux, model_type, node, arguments):
    widgets = {"fov (DSINE)": [30.0], "iterations (DSINE)": [7], "backbone (Metric3D)": ["vit-large"], "fx (Metric3D)": [800], "fy (Metric3D)": [600]}
    [[preprocessor]] = run(pack.segs_support.NormalMap_Preprocessor_Provider_for_SEGS, model_type=[model_type], **widgets)
    assert preprocessor.apply(picture()) is OUTPUT
    assert aux == [(node, {**arguments, "resolution": 128})]


def test_normal_map_widgets_left_out_of_a_prompt_take_their_defaults(pack, run, aux):
    [[preprocessor]] = run(pack.segs_support.NormalMap_Preprocessor_Provider_for_SEGS, model_type=["Metric3D"])
    preprocessor.apply(picture())
    assert aux == [("Metric3D-NormalMapPreprocessor", {"backbone": "vit-small", "fx": 1000, "fy": 1000, "resolution": 128})]


def test_depth_anything_offers_the_files_of_both_versions(pack):
    files, _ = pack.segs_support.DepthAnything_Preprocessor_Provider_for_SEGS.INPUT_TYPES()["required"]["ckpt_name"]
    assert sorted(files) == sorted(V1_FILES + V2_FILES)


def test_depth_anything_takes_the_version_from_the_file_unless_one_is_chosen(pack):
    versions, options = pack.segs_support.DepthAnything_Preprocessor_Provider_for_SEGS.INPUT_TYPES()["required"]["version"]
    assert versions == ["auto", "v1", "v2"] and options["default"] == "auto"


@pytest.mark.parametrize("ckpt_name,node", [(name, "DepthAnythingPreprocessor") for name in V1_FILES]
                         + [(name, "DepthAnythingV2Preprocessor") for name in V2_FILES])
def test_depth_anything_on_auto_runs_the_node_of_the_version_its_file_belongs_to(pack, run, aux, caplog, ckpt_name, node):
    [[preprocessor]] = run(pack.segs_support.DepthAnything_Preprocessor_Provider_for_SEGS, version=["auto"], ckpt_name=[ckpt_name])
    assert preprocessor.apply(picture()) is OUTPUT
    assert aux == [(node, {"ckpt_name": ckpt_name, "resolution": 128})]
    assert not caplog.records


@pytest.mark.parametrize("version,ckpt_name,node", [
    ("v1", "depth_anything_v2_vitl.pth", "DepthAnythingPreprocessor"),
    ("v2", "depth_anything_vitl14.pth", "DepthAnythingV2Preprocessor"),
    ("v1", "renamed.pth", "DepthAnythingPreprocessor"),
])
def test_depth_anything_runs_the_node_of_a_chosen_version_whatever_the_file_is_called(pack, run, aux, caplog, version, ckpt_name, node):
    [[preprocessor]] = run(pack.segs_support.DepthAnything_Preprocessor_Provider_for_SEGS, version=[version], ckpt_name=[ckpt_name])
    preprocessor.apply(picture())
    assert aux == [(node, {"ckpt_name": ckpt_name, "resolution": 128})]
    assert not caplog.records


def test_depth_anything_on_auto_warns_about_a_file_whose_name_tells_no_version_and_runs_v2(pack, run, aux, caplog):
    [[preprocessor]] = run(pack.segs_support.DepthAnything_Preprocessor_Provider_for_SEGS, version=["auto"], ckpt_name=["renamed.pth"])
    assert [record.levelname for record in caplog.records] == ["WARNING"] and "renamed.pth" in caplog.text
    preprocessor.apply(picture())
    assert aux == [("DepthAnythingV2Preprocessor", {"ckpt_name": "renamed.pth", "resolution": 128})]


@pytest.mark.parametrize("provider,inputs", [
    ("NormalMap_Preprocessor_Provider_for_SEGS", {"model_type": ["DSINE"]}),
    ("NormalMap_Preprocessor_Provider_for_SEGS", {"model_type": ["BAE"]}),
    ("NormalMap_Preprocessor_Provider_for_SEGS", {"model_type": ["Metric3D"]}),
    ("DepthAnything_Preprocessor_Provider_for_SEGS", {"version": ["auto"], "ckpt_name": ["depth_anything_vitl14.pth"]}),
    ("DepthAnything_Preprocessor_Provider_for_SEGS", {"version": ["auto"], "ckpt_name": ["depth_anything_v2_vitl.pth"]}),
])
def test_without_controlnet_aux_the_preprocessor_asks_to_install_it_and_stops(pack, run, monkeypatch, provider, inputs):
    import nodes
    for name in AUX_NODES:
        monkeypatch.delitem(nodes.NODE_CLASS_MAPPINGS, name, raising=False)
    asked = []
    monkeypatch.setattr(pack.utils, "try_install_custom_node", lambda url, msg: asked.append(url))
    [[preprocessor]] = run(getattr(pack.segs_support, provider), **inputs)
    with pytest.raises(Exception, match="ControlNet Auxiliary Preprocessors"):
        preprocessor.apply(picture())
    assert asked == [AUX_URL]
