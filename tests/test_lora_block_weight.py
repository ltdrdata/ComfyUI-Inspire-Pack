from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SIZE, RANK = 4, 2
SD15_BLOCKS_IN_VECTOR_ORDER = ([f"input_blocks.{n}.1.proj_in" for n in (1, 2, 4, 5, 7, 8)] + ["middle_block.1.proj_in"]
                               + [f"output_blocks.{n}.1.proj_in" for n in range(3, 12)])
OUTSIDE = "out.2"
TEXT = "clip_l.transformer.text_model.encoder.layers.0.mlp.fc1"
ONES = ",".join(["1"] * 17)


def module(keys):
    root = torch.nn.Module()
    for key in keys:
        node = root
        *path, last = key.split(".")
        for name in path:
            if name not in node._modules:
                node.add_module(name, torch.nn.Module())
            node = node._modules[name]
        node.add_module(last, torch.nn.Linear(SIZE, SIZE, bias=False))
    return root


@pytest.fixture
def tiny(pack):
    import comfy.model_patcher
    import comfy.sd
    torch.manual_seed(0)
    cpu = torch.device("cpu")
    unet = module(["diffusion_model." + key for key in SD15_BLOCKS_IN_VECTOR_ORDER + [OUTSIDE]])
    unet.model_config = SimpleNamespace(unet_config={})
    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = module([TEXT])
    clip.patcher = comfy.model_patcher.ModelPatcher(clip.cond_stage_model, cpu, cpu)
    clip.tokenizer, clip.tokenizer_metadata, clip.tokenizer_options = None, {}, {}
    clip.layer_idx, clip.use_clip_schedule, clip.apply_hooks_to_conds = None, False, None
    lora = {}
    for key, name in [(key, "lora_unet_" + key.replace(".", "_")) for key in SD15_BLOCKS_IN_VECTOR_ORDER + [OUTSIDE]] + [(TEXT, "lora_te_text_model_encoder_layers_0_mlp_fc1")]:
        lora[name + ".lora_up.weight"] = torch.randn(SIZE, RANK)
        lora[name + ".lora_down.weight"] = torch.randn(RANK, SIZE)
    return SimpleNamespace(model=comfy.model_patcher.ModelPatcher(unet, cpu, cpu), clip=clip, lora=lora)


def weights(model, clip):
    found = {}
    for patcher, prefix in ((model, "diffusion_model."), (clip.patcher, "")):
        patcher.patch_model()
        found.update({key[len(prefix):-len(".weight")]: value.clone() for key, value in patcher.model.state_dict().items()})
        patcher.unpatch_model()
    return found


def comfy_lora(tiny, strength_model=0.8, strength_clip=0.5):
    import comfy.sd
    return weights(*comfy.sd.load_lora_for_models(tiny.model, tiny.clip, tiny.lora, strength_model, strength_clip))


def lbw(pack, tiny, vector, strength_model=0.8, strength_clip=0.5, inverse=False, seed=0, A=4.0, B=1.0):
    model, clip, populated = pack.lora_block_weight.LoraLoaderBlockWeight.load_lora_for_models(
        tiny.model, tiny.clip, tiny.lora, strength_model, strength_clip, inverse, seed, A, B, vector)
    return weights(model, clip), populated


def test_a_vector_of_ones_patches_as_comfyuis_own_lora_loader(pack, tiny):
    got, populated = lbw(pack, tiny, ONES)
    expected = comfy_lora(tiny)
    assert populated == ONES
    assert got.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(got[key], expected[key])
    base = weights(tiny.model, tiny.clip)
    assert all(not torch.equal(base[key], expected[key]) for key in expected)


def test_a_block_at_zero_keeps_its_weight_and_a_block_at_a_half_gets_half_the_strength(pack, tiny):
    vector = ["1"] * 17
    vector[1], vector[7], vector[16] = "0", "0", "0.5"
    got, populated = lbw(pack, tiny, ",".join(vector))
    assert populated == ",".join(vector)
    base, full, half = weights(tiny.model, tiny.clip), comfy_lora(tiny), comfy_lora(tiny, 0.4, 0.25)
    assert torch.equal(got[SD15_BLOCKS_IN_VECTOR_ORDER[0]], base[SD15_BLOCKS_IN_VECTOR_ORDER[0]]) and torch.equal(got[SD15_BLOCKS_IN_VECTOR_ORDER[6]], base[SD15_BLOCKS_IN_VECTOR_ORDER[6]])
    torch.testing.assert_close(got[SD15_BLOCKS_IN_VECTOR_ORDER[15]], half[SD15_BLOCKS_IN_VECTOR_ORDER[15]])
    for key in SD15_BLOCKS_IN_VECTOR_ORDER[1:6] + SD15_BLOCKS_IN_VECTOR_ORDER[7:15] + [OUTSIDE, TEXT]:
        torch.testing.assert_close(got[key], full[key])


def test_the_first_value_weights_the_text_encoder_and_the_weights_outside_the_blocks(pack, tiny):
    got, _ = lbw(pack, tiny, "0," + ",".join(["1"] * 16))
    base, full = weights(tiny.model, tiny.clip), comfy_lora(tiny)
    assert torch.equal(got[TEXT], base[TEXT]) and torch.equal(got[OUTSIDE], base[OUTSIDE])
    for key in SD15_BLOCKS_IN_VECTOR_ORDER:
        torch.testing.assert_close(got[key], full[key])


def test_a_and_b_stand_for_their_values_and_lower_case_for_half_of_them(pack, tiny):
    got, populated = lbw(pack, tiny, "A,a,B,b," + ",".join(["0"] * 13), A=3.0, B=1.0)
    assert populated == "3.0,1.5,1,0.5," + ",".join(["0"] * 13)
    torch.testing.assert_close(got[SD15_BLOCKS_IN_VECTOR_ORDER[0]], comfy_lora(tiny, 0.8 * 1.5)[SD15_BLOCKS_IN_VECTOR_ORDER[0]])
    torch.testing.assert_close(got[TEXT], comfy_lora(tiny, 0.8, 0.5 * 3.0)[TEXT])


def test_r_draws_values_from_the_seed(pack, tiny):
    _, first = lbw(pack, tiny, "1," + ",".join(["R"] * 16), seed=7)
    _, again = lbw(pack, tiny, "1," + ",".join(["R"] * 16), seed=7)
    _, other = lbw(pack, tiny, "1," + ",".join(["R"] * 16), seed=8)
    assert first == again and first != other
    assert all(0 <= float(value) <= 3 for value in first.split(",")[1:])


def test_inverse_applies_one_minus_each_value(pack, tiny):
    got, populated = lbw(pack, tiny, ",".join(["0"] * 17), inverse=True)
    assert populated == ONES
    expected = comfy_lora(tiny)
    for key in expected:
        torch.testing.assert_close(got[key], expected[key])


@pytest.mark.parametrize("vector,populated", [
    ("SD-INS", "1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0"),
    ("SD-OUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1", "1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1"),
    ("%default=0,in=1", "0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0"),
    ("%mid=0", "1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1"),
    ("%base=0", "0," + ",".join(["1"] * 16)),
])
def test_presets_and_block_specs_give_a_vector(pack, tiny, vector, populated):
    assert lbw(pack, tiny, vector)[1] == populated


@pytest.mark.parametrize("vector", ["1,0,x,1,1,1,1,1,1,1,1,1,1,1,1,1,1", "1,0,1", "NO-SUCH-PRESET"])
def test_a_vector_it_cannot_read_is_refused(pack, tiny, vector):
    with pytest.raises(ValueError, match="invalid block_vector"):
        lbw(pack, tiny, vector)


def presets():
    return [line.strip() for line in (ROOT / "resources" / "lbw-preset.txt").read_text().splitlines()]


def test_the_preset_combo_lists_the_shipped_presets(pack):
    lbw_module = pack.lora_block_weight
    plain = ["Preset"] + [line for line in presets() if not line.startswith("@")]
    assert lbw_module.LoraLoaderBlockWeight.INPUT_TYPES()["required"]["preset"][0] == plain
    assert lbw_module.MakeLBW.INPUT_TYPES()["required"]["preset"][0] == plain
    assert lbw_module.XYInput_LoraBlockWeight.INPUT_TYPES()["required"]["preset"][0] == ["Preset"] + presets()


@pytest.fixture
def lora_file(pack, tiny, tmp_path, monkeypatch):
    import comfy.utils
    comfy.utils.save_torch_file(tiny.lora, str(tmp_path / "unit-tiny.safetensors"))
    monkeypatch.setitem(pack.folder_paths.folder_names_and_paths, "loras", ([str(tmp_path)], {".safetensors"}))
    return "unit-tiny.safetensors"


def test_make_and_apply_lbw_give_what_the_loader_gives(pack, tiny, lora_file):
    vector = "1,0,1,1,1,1,1,1,0.5,1,1,1,1,1,1,1,0"
    nodes = pack.lora_block_weight
    lbw_model, populated = nodes.MakeLBW().doit(tiny.model, tiny.clip, lora_file, False, 0, 4.0, 1.0, "Preset", vector)
    applied = weights(*nodes.ApplyLBW.doit(tiny.model, tiny.clip, 0.8, 0.5, lbw_model))
    model, clip, loaded = nodes.LoraLoaderBlockWeight().doit(tiny.model, tiny.clip, lora_file, 0.8, 0.5, False, 0, 4.0, 1.0, "Preset", vector)
    assert populated == loaded == vector
    expected = weights(model, clip)
    for key in expected:
        torch.testing.assert_close(applied[key], expected[key])


def test_the_loader_passes_the_model_on_when_bypassed_or_at_zero_strength(pack, tiny, lora_file):
    loader = pack.lora_block_weight.LoraLoaderBlockWeight()
    assert loader.doit(tiny.model, tiny.clip, lora_file, 1.0, 1.0, False, 0, 4.0, 1.0, "Preset", ONES, bypass=True) == (tiny.model, tiny.clip, "")
    assert loader.doit(tiny.model, tiny.clip, lora_file, 0.0, 0.0, False, 0, 4.0, 1.0, "Preset", ONES) == (tiny.model, tiny.clip, "")


def test_lora_block_info_counts_the_weights_of_each_block(pack, tiny, lora_file):
    pack.lora_block_weight.LoraBlockInfo().doit(tiny.model, tiny.clip, lora_file, "", "5")
    ((event, data),) = pack.prompt_server.sent
    assert (event, data["node_id"], data["widget_name"]) == ("inspire-node-feedback", "5", "block_info")
    text = data["data"]
    assert "[Input blocks] (6, Subs=6)" in text and " IN8: 1\n" in text
    assert "[Middle blocks] (1, Subs=1)" in text and " MID1: 1\n" in text
    assert "[Output blocks] (9, Subs=9)" in text and " OUT11: 1\n" in text
    assert "[Base blocks] (2, Subs=2)" in text and " TXT_ENC0: 1\n" in text and " out.2.weight\n" in text
