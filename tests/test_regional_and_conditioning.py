import copy

import pytest
import torch


def color_image():
    image = torch.zeros(1, 2, 16, 3)
    image[0, :, 0:4] = torch.tensor([1.0, 0.0, 0.0])
    image[0, :, 4:8] = torch.tensor([0.0, 1.0, 0.0])
    image[0, :, 8:12] = 1.0
    return image


def columns(*spans, width=16, height=2):
    mask = torch.zeros(1, height, width)
    for start, stop, value in spans:
        mask[:, :, start:stop] = value
    return mask


@pytest.mark.parametrize("color", ["#FF0000", "red", "16711680"])
def test_color_to_mask_takes_a_hex_code_a_name_or_a_number(pack, color):
    assert torch.equal(pack.regional_nodes.color_to_mask(color_image(), color), columns((0, 4, 1.0)))


def test_color_to_mask_refuses_what_is_not_a_color(pack):
    with pytest.raises(Exception, match="Invalid mask_color"):
        pack.regional_nodes.color_to_mask(color_image(), "#XYZ")


@pytest.mark.parametrize("method,expected", [
    ("override", columns((0, 4, 1.0), (4, 8, 0.5), (8, 16, 0.2))),
    ("max", columns((0, 4, 1.0), (4, 8, 0.5), (8, 16, 0.2))),
    ("sum", columns((0, 4, 1.0), (4, 8, 0.7), (8, 16, 0.2))),
])
def test_color_mask_to_depth_mask_flattens_the_layers_over_the_base_value(pack, method, expected):
    spec = "#FF0000:1.0\n#00FF00:0.5"
    (mask,) = pack.regional_nodes.ColorMaskToDepthMask().doit(color_image(), spec, 0.2, 0, method)
    torch.testing.assert_close(mask, expected)


def test_color_mask_to_depth_mask_override_lets_an_upper_layer_win(pack):
    image = color_image()
    (mask,) = pack.regional_nodes.ColorMaskToDepthMask().doit(image, "#FFFFFF:0.3\n#FFFFFF:0.9", 0.0, 0, "override")
    torch.testing.assert_close(mask, columns((8, 12, 0.9)))


def test_color_mask_to_depth_mask_dilates_each_layer(pack):
    (mask,) = pack.regional_nodes.ColorMaskToDepthMask().doit(color_image(), "#00FF00:1.0", 0.0, 1, "max")
    torch.testing.assert_close(mask, columns((3, 9, 1.0)))


def noise():
    return torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))


def variation(seed):
    return torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(seed))


def test_regional_seed_explorer_varies_the_noise_inside_the_mask_only(pack):
    mask = torch.zeros(1, 64, 64)
    mask[:, :, :32] = 1.0
    start = noise()
    (out,) = pack.regional_nodes.RegionalSeedExplorerMask.doit(mask, start.clone(), "", True, 9, 1.0, "CPU")
    torch.testing.assert_close(out[..., :4], variation(9)[..., :4])
    assert torch.equal(out[..., 4:], start[..., 4:])
    (half,) = pack.regional_nodes.RegionalSeedExplorerMask.doit(mask, start.clone(), "9:0.3", False, 0, 0.0, "CPU")
    assert not torch.allclose(half[..., :4], start[..., :4]) and torch.equal(half[..., 4:], start[..., 4:])


def test_regional_seed_explorer_by_color_mask_varies_the_region_of_its_color(pack):
    image = torch.zeros(1, 64, 64, 3)
    image[0, :, 32:] = 1.0
    start = noise()
    out, mask = pack.regional_nodes.RegionalSeedExplorerColorMask.doit(image, "#FFFFFF", start.clone(), "", True, 9, 1.0, "CPU")
    assert mask.shape == (1, 64, 64) and bool(mask[0, :, 32:].all()) and not mask[0, :, :32].any()
    torch.testing.assert_close(out[..., 4:], variation(9)[..., 4:])
    assert torch.equal(out[..., :4], start[..., :4])


def test_regional_cfg_applies_the_cfg_inside_the_mask_and_the_uncond_outside(pack):
    import comfy.model_patcher
    import comfy.samplers
    cpu = torch.device("cpu")
    model = comfy.model_patcher.ModelPatcher(torch.nn.Linear(1, 1), cpu, cpu)
    mask = torch.zeros(16, 16)
    mask[:, :8] = 1.0
    (patched,) = pack.regional_nodes.RegionalCFG.doit(model, mask)
    cond, uncond, x = torch.full((1, 4, 4, 4), 2.0), torch.full((1, 4, 4, 4), 0.5), torch.zeros(1, 4, 4, 4)
    out = comfy.samplers.cfg_function(patched.model, cond, uncond, 7.0, x, torch.ones(1), model_options=patched.model_options)
    assert torch.equal(out[..., :2], torch.full((1, 4, 4, 2), 0.5 + 1.5 * 7.0))
    assert torch.equal(out[..., 2:], torch.full((1, 4, 4, 2), 0.5))
    assert "sampler_cfg_function" not in model.model_options


def test_apply_regional_ipadapters_applies_each_entry_with_the_models_of_the_pipe(pack, monkeypatch):
    import nodes
    calls = []

    def stand_in(name):
        class Node:
            def apply_ipadapter(self, model, **inputs):
                calls.append((name, model, inputs))
                return (model + (name,),)
        return Node

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "IPAdapterAdvanced", stand_in("advanced"))
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "IPAdapterEmbeds", stand_in("embeds"))
    regional = pack.regional_nodes
    (by_image,) = regional.RegionalIPAdapterMask.doit("left", "picture", 0.7, 0.5, "linear", 0.1, 0.9, False, False, 1.2, "average", "negative")
    (by_embeds,) = regional.RegionalIPAdapterEncodedMask.doit("right", "embeds", 0.4, "ease in", 0.2, 0.8, False, "negative embeds")
    pipe = ("IPADAPTER", ("base",), "CLIP_VISION", "INSIGHTFACE", lambda model: model)
    (model,) = regional.ApplyRegionalIPAdapters.doit(ipadapter_pipe=pipe, regional_ipadapter1=by_image, regional_ipadapter2=by_embeds)
    assert model == ("base", "advanced", "embeds")
    assert calls == [
        ("advanced", ("base",), dict(ipadapter="IPADAPTER", weight=0.7, weight_type="linear", start_at=0.1, end_at=0.9, combine_embeds="average",
                                     clip_vision="CLIP_VISION", image="picture", image_negative="negative", attn_mask="left",
                                     insightface="INSIGHTFACE", weight_faceidv2=1.2)),
        ("embeds", ("base", "advanced"), dict(ipadapter="IPADAPTER", pos_embed="embeds", weight=0.4, weight_type="ease in", start_at=0.2, end_at=0.8,
                                              neg_embed="negative embeds", attn_mask="right", clip_vision="CLIP_VISION")),
    ]


def test_apply_regional_adapters_applies_the_model_parts_in_the_order_of_its_inputs_and_passes_the_prompts_on(pack, comfyui, monkeypatch):
    import nodes

    class Model:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {}}
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "doit"

        def doit(self):
            return (("base",),)

    class Prompt:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"text": ("STRING", {"default": ""})}}
        RETURN_TYPES = ("CONDITIONING",)
        FUNCTION = "doit"

        def doit(self, text):
            return ([[text, {}]],)

    class Entry:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"name": ("STRING", {"default": ""})}}
        RETURN_TYPES = ("REGIONAL_ADAPTER",)
        FUNCTION = "doit"

        def doit(self, name):
            return (pack.regional_nodes.RegionalAdapter(None, apply_model=lambda model: model + (name,)),)

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "UnitModel", Model)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "UnitPrompt", Prompt)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "UnitRegionalAdapter", Entry)
    prompt = {
        "1": {"class_type": "UnitModel", "inputs": {}},
        "2": {"class_type": "UnitRegionalAdapter", "inputs": {"name": "face of the left person"}},
        "3": {"class_type": "UnitRegionalAdapter", "inputs": {"name": "face of the right person"}},
        "4": {"class_type": "UnitRegionalAdapter", "inputs": {"name": "background"}},
        "5": {"class_type": "UnitPrompt", "inputs": {"text": "two people in a park"}},
        "6": {"class_type": "UnitPrompt", "inputs": {"text": "blurry"}},
        "7": {"class_type": "ApplyRegionalAdapters //Inspire", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "regional_adapter1": ["3", 0],
                                                                          "regional_adapter2": ["2", 0], "regional_adapter3": ["4", 0]}},
    }
    for index, key in enumerate(("unit model", "unit positive", "unit negative")):
        prompt[str(8 + index)] = {"class_type": "CacheBackendData //Inspire", "inputs": {"key": key, "tag": "unit", "data": ["7", index]}}
    comfyui.execute(prompt, ["8", "9", "10"])
    cached = {key: data for key, (_, (_, data)) in pack.backend_support.cache.items()}
    assert cached == {"unit model": ("base", "face of the right person", "face of the left person", "background"),
                      "unit positive": [["two people in a park", {}]], "unit negative": [["blurry", {}]]}


def region(pack, name, mask, changes_prompt=True):
    def apply_conditioning(positive, negative):
        return [[c, d | {"region": name, "mask": mask}] for c, d in positive], [[c, d | {"region": name}] for c, d in negative]
    return pack.regional_nodes.RegionalAdapter(mask, apply_model=lambda model: model + (name,), apply_conditioning=apply_conditioning if changes_prompt else None)


def test_apply_regional_adapters_gives_each_region_its_copy_of_the_prompt_and_the_plain_prompt_where_no_such_region_lies(pack):
    left = columns((0, 24, 1.0), width=64, height=64)
    middle = columns((24, 40, 1.0), width=64, height=64)
    right_at_half_size = columns((20, 32, 1.0), width=32, height=32)
    positive, negative = [[torch.ones(1), {"text": "two people in a park"}]], [[torch.zeros(1), {"text": "blurry"}]]
    model, positives, negatives = pack.regional_nodes.ApplyRegionalAdapters.doit(
        ("base",), positive, negative, regional_adapter1=region(pack, "left", left), regional_adapter2=region(pack, "middle", middle, changes_prompt=False),
        regional_adapter3=region(pack, "right", right_at_half_size))
    assert model == ("base", "left", "middle", "right")
    assert [(d.get("region"), d["text"]) for _, d in positives] == [("left", "two people in a park"), ("right", "two people in a park"), (None, "two people in a park")]
    assert [d["region"] for _, d in negatives] == ["left", "right"]
    rest = positives[2][1]["mask"]
    assert rest.shape[-2:] == (64, 64)
    assert (rest[..., :24] == 0).all() and (rest[..., 24:39] == 1).all() and (rest[..., 41:] == 0).all()
    assert positive == [[torch.ones(1), {"text": "two people in a park"}]]


def test_apply_regional_adapters_adds_no_plain_prompt_when_the_regions_cover_the_picture(pack):
    positive, negative = [[torch.ones(1), {"text": "two people in a park"}]], [[torch.zeros(1), {"text": "blurry"}]]
    _, positives, _ = pack.regional_nodes.ApplyRegionalAdapters.doit(
        ("base",), positive, negative, regional_adapter1=region(pack, "left", columns((0, 8, 1.0))), regional_adapter2=region(pack, "right", columns((8, 16, 1.0))))
    assert [d["region"] for _, d in positives] == ["left", "right"]


@pytest.mark.parametrize("positive,negative", [(None, None), ([[torch.ones(1), {}]], None), (None, [[torch.ones(1), {}]])])
def test_apply_regional_adapters_asks_for_both_prompts_before_it_applies_an_entry_that_changes_them(pack, positive, negative):
    applied = []
    entry = pack.regional_nodes.RegionalAdapter(columns((0, 8, 1.0)), apply_model=applied.append, apply_conditioning=lambda positive, negative: (positive, negative))
    with pytest.raises(Exception, match="link `positive` and `negative`"):
        pack.regional_nodes.ApplyRegionalAdapters.doit(("base",), positive, negative, regional_adapter1=entry)
    assert applied == []


def test_regional_pulid_applies_apply_pulid_with_its_mask_and_settings_in_apply_regional_adapters(pack, monkeypatch):
    import nodes
    calls = []

    class ApplyPulid:
        def apply_pulid(self, model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None):
            calls.append((model, pulid, eva_clip, face_analysis, image, weight, start_at, end_at, method, attn_mask))
            return (model + (image,),)

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "ApplyPulid", ApplyPulid)
    regional = pack.regional_nodes
    (left,) = regional.RegionalPulid.doit("PULID", "EVA_CLIP", "FACEANALYSIS", "left mask", "face A", "fidelity", 0.8, 0.1, 0.9)
    (right,) = regional.RegionalPulid.doit("PULID", "EVA_CLIP", "FACEANALYSIS", "right mask", "face B", "style", 1.2, 0.0, 1.0)
    assert regional.ApplyRegionalAdapters.doit(("base",), "positive", "negative", regional_adapter1=left, regional_adapter2=right) == (
        ("base", "face A", "face B"), "positive", "negative")
    assert calls == [(("base",), "PULID", "EVA_CLIP", "FACEANALYSIS", "face A", 0.8, 0.1, 0.9, "fidelity", "left mask"),
                     (("base", "face A"), "PULID", "EVA_CLIP", "FACEANALYSIS", "face B", 1.2, 0.0, 1.0, "style", "right mask")]


@pytest.fixture
def instantid_nodes(pack, monkeypatch):
    import nodes
    calls = []

    class InstantIDAttentionPatch:
        def patch_attention(self, instantid, insightface, image, model, weight, start_at, end_at, noise=0.0, mask=None):
            if model is not None:
                calls.append(("attention", image, weight, start_at, end_at, noise, mask))
                model = model + (image,)
            return (model, {"cond": image, "noise": noise})

    class FaceKeypointsPreprocessor:
        def preprocess_image(self, faceanalysis, image):
            calls.append(("keypoints", image))
            return ("keypoints of the placed reference",)

    class ApplyInstantIDControlNet:
        def apply_controlnet(self, face_embeds, control_net, image_kps, positive, negative, strength, start_at, end_at, mask=None):
            calls.append(("controlnet", face_embeds, image_kps, strength, start_at, end_at, mask))
            return ([[c, d | {"face": face_embeds["cond"], "mask": mask}] for c, d in positive], [[c, d | {"face": face_embeds["cond"]}] for c, d in negative])

    for node in (InstantIDAttentionPatch, FaceKeypointsPreprocessor, ApplyInstantIDControlNet):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node.__name__, node)
    return calls


def test_regional_instantid_patches_the_model_and_gives_its_region_a_copy_of_the_prompt_with_the_controlnet(pack, instantid_nodes):
    regional = pack.regional_nodes
    left, right = columns((0, 6, 1.0)), columns((10, 16, 1.0))
    (a,) = regional.RegionalInstantID.doit("INSTANTID", "FACEANALYSIS", "CONTROL_NET", left, "face A", 0.8, 0.6, 0.1, 0.9, 0.2, image_kps="pose A")
    (b,) = regional.RegionalInstantID.doit("INSTANTID", "FACEANALYSIS", "CONTROL_NET", right, "face B", 1.2, 1.5, 0.0, 1.0, 0.0, image_kps="pose B")
    positive, negative = [[torch.ones(1), {"text": "two people in a park"}]], [[torch.zeros(1), {"text": "blurry"}]]
    model, positives, negatives = regional.ApplyRegionalAdapters.doit(("base",), positive, negative, regional_adapter1=a, regional_adapter2=b)
    assert model == ("base", "face A", "face B")
    assert [(d.get("face"), d["text"]) for _, d in positives] == [("face A", "two people in a park"), ("face B", "two people in a park"), (None, "two people in a park")]
    assert [d["face"] for _, d in negatives] == ["face A", "face B"]
    assert torch.equal(positives[2][1]["mask"], columns((6, 10, 1.0)))
    assert [call[:-1] for call in instantid_nodes] == [
        ("attention", "face A", 0.8, 0.1, 0.9, 0.2), ("attention", "face B", 1.2, 0.0, 1.0, 0.0),
        ("controlnet", {"cond": "face A", "noise": 0.2}, "pose A", 0.6, 0.1, 0.9), ("controlnet", {"cond": "face B", "noise": 0.0}, "pose B", 1.5, 0.0, 1.0)]
    assert [call[-1] for call in instantid_nodes] == [left, right, left, right]


def test_regional_instantid_places_the_reference_in_its_mask_for_the_keypoints_when_none_are_linked(pack, instantid_nodes):
    region = columns((8, 16, 1.0), width=32, height=24)
    (entry,) = pack.regional_nodes.RegionalInstantID.doit("INSTANTID", "FACEANALYSIS", "CONTROL_NET", region, torch.ones(1, 8, 8, 3), 0.8, 0.8, 0.0, 1.0, 0.0)
    entry.apply_conditioning([[torch.ones(1), {}]], [[torch.zeros(1), {}]])
    placed = torch.zeros(1, 24, 32, 3)
    placed[:, 4:12, 8:16] = 1.0
    assert [torch.equal(call[1], placed) for call in instantid_nodes if call[0] == "keypoints"] == [True]
    assert [call[2] for call in instantid_nodes if call[0] == "controlnet"] == ["keypoints of the placed reference"]


def conditioning():
    return [[torch.ones(1, 2, 3), {"area": (8, 16, 0, 8), "strength": 1.0, "control": "net", "control_apply_to_uncond": True}],
            [torch.zeros(1, 2, 3), {"pooled_output": "pooled"}]]


def test_remove_controlnet_takes_the_controlnet_off_each_conditioning(pack):
    given = conditioning()
    before = copy.deepcopy(given)
    (out,) = pack.prompt_support.RemoveControlNet().doit(given)
    assert [entry[1] for entry in out] == [{"area": (8, 16, 0, 8), "strength": 1.0}, {"pooled_output": "pooled"}]
    assert out[0][0] is given[0][0]
    assert [entry[1] for entry in given] == [entry[1] for entry in before]


def test_conditioning_upscale_scales_the_area_and_rounds_it_up_to_eight(pack):
    given = conditioning()
    (out,) = pack.conditioning_nodes.ConditioningUpscale.upscale(given, 2)
    assert out[0][1]["area"] == (16, 32, 0, 16)
    assert pack.conditioning_nodes.ConditioningUpscale.upscale([[torch.ones(1), {"area": (3, 5, 1, 2)}]], 3)[0][0][1]["area"] == (16, 16, 8, 8)
    assert out[1][1] == {"pooled_output": "pooled"}
    assert given[0][1]["area"] == (8, 16, 0, 8)


def test_conditioning_stretch_moves_the_area_from_one_resolution_to_another(pack):
    given = conditioning()
    (out,) = pack.conditioning_nodes.ConditioningStretch.upscale(given, 512, 512, 1024, 768)
    assert out[0][1]["area"] == (16, 32, 0, 16)
    assert given[0][1]["area"] == (8, 16, 0, 8)
    assert out[1][1] == {"pooled_output": "pooled"}
