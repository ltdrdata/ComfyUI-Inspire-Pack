import pytest
import torch
from torch.testing import assert_close


def latent(batch=1, height=8, width=12):
    return {"samples": torch.zeros(batch, 4, height, width)}


def core(seed, batch=1, batch_index=None):
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
    samples = latent(batch)
    if batch_index is not None:
        samples["batch_index"] = batch_index
    return Noise_RandomNoise(seed).generate_noise(samples)


def inspire(pack, seed, batch=1, mode="comfy", strength=0.0, variation_seed=0, method="linear", **extra):
    noise = pack.a1111_compat.RandomNoise().get_noise(seed, "CPU", mode, variation_seed, strength, method)[0]
    return noise.generate_noise({**latent(batch), **extra})


@pytest.mark.parametrize("batch", [1, 3])
def test_comfy_batch_mode_on_the_cpu_is_the_noise_of_comfyui(pack, batch):
    assert torch.equal(inspire(pack, 5, batch), core(5, batch))


def test_comfy_batch_mode_follows_the_batch_index_as_comfyui_does(pack):
    assert torch.equal(inspire(pack, 5, 3, batch_index=[2, 0, 2]), core(5, 3, [2, 0, 2]))


def test_incremental_batch_mode_gives_each_picture_the_noise_of_the_next_seed(pack):
    noise = inspire(pack, 5, 3, "incremental")
    for index in range(3):
        assert torch.equal(noise[index:index + 1], core(5 + index))


def test_the_same_seed_gives_the_same_noise_and_another_seed_another(pack):
    assert torch.equal(inspire(pack, 5, 2, "incremental"), inspire(pack, 5, 2, "incremental"))
    assert not torch.equal(inspire(pack, 5), inspire(pack, 6))


@pytest.mark.parametrize("method", ["linear", "slerp"])
def test_variation_strength_moves_the_noise_from_the_seed_to_the_variation_seed(pack, method):
    assert torch.equal(inspire(pack, 5, 2, strength=0.0, variation_seed=9, method=method), core(5, 2))
    full = inspire(pack, 5, 2, strength=1.0, variation_seed=9, method=method)
    assert_close(full, core(9).expand(2, -1, -1, -1))
    part = inspire(pack, 5, 2, strength=0.3, variation_seed=9, method=method)
    assert not torch.allclose(part, core(5, 2)) and not torch.allclose(part, full)
    assert part.std().item() == pytest.approx(1.0, abs=0.15)


def test_variation_str_inc_raises_the_strength_picture_by_picture(pack):
    noise = inspire(pack, 5, 3, "variation str inc:0.05", strength=0.1, variation_seed=9)
    for index in range(3):
        assert_close(noise[index:index + 1], inspire(pack, 5, 1, strength=0.1 + 0.05 * index, variation_seed=9))


def test_internal_seed_is_the_seed_the_sampler_gets_and_noise_seed_makes_the_noise(pack):
    node = pack.a1111_compat.RandomNoise()
    plain = node.get_noise(5, "CPU", "comfy", 0, 0.0)[0]
    assert plain.seed == 5
    other = node.get_noise(5, "CPU", "comfy", 0, 0.0, internal_seed=9)[0]
    assert other.seed == 9
    assert torch.equal(other.generate_noise(latent()), plain.generate_noise(latent()))


def explore(pack, prompt, enable_additional=False, seed=0, strength=0.0, mode="comfy", batch=1):
    return pack.prompt_support.SeedExplorer.doit(latent(batch), prompt, enable_additional, seed, strength, "CPU", mode)[0]


def test_seed_explorer_starts_from_the_noise_of_its_first_seed(pack):
    assert torch.equal(explore(pack, "5"), core(5))
    noise = explore(pack, "5", mode="incremental", batch=2)
    assert torch.equal(noise[1:], core(6))


def test_seed_explorer_moves_toward_each_further_seed_by_its_strength(pack):
    assert torch.equal(explore(pack, "5, 9:0"), core(5))
    assert_close(explore(pack, "5,\n9:1.0"), core(9))
    assert not torch.allclose(explore(pack, "5, 9:0.3"), core(5))
    assert torch.equal(explore(pack, "5, 9:0.3"), explore(pack, "5", True, 9, 0.3))


def test_seed_explorer_skips_an_entry_it_cannot_read(pack):
    assert torch.equal(explore(pack, "5, nine:0.3"), core(5))


def composite(pack, mode, x=0, y=0, size=8):
    destination = torch.zeros(1, 4, size, size)
    source = torch.ones(1, 4, 4, 4)
    return pack.prompt_support.CompositeNoise().doit(destination, source, mode, x, y)[0], destination


@pytest.mark.parametrize("mode,x,y,top,left", [
    ("center", 0, 0, 2, 2), ("left-top", 0, 0, 0, 0), ("right-top", 0, 0, 0, 4),
    ("left-bottom", 0, 0, 4, 0), ("right-bottom", 0, 0, 4, 4), ("xy", 1, 3, 3, 1),
])
def test_composite_noise_lays_the_source_on_the_destination(pack, mode, x, y, top, left):
    out, destination = composite(pack, mode, x, y)
    expected = torch.zeros(1, 4, 8, 8)
    expected[:, :, top:top + 4, left:left + 4] = 1
    assert torch.equal(out, expected)
    assert not destination.any()


def test_composite_noise_cuts_off_what_falls_outside_the_destination(pack):
    out, _ = composite(pack, "xy", 6, 7)
    expected = torch.zeros(1, 4, 8, 8)
    expected[:, :, 7:, 6:] = 1
    assert torch.equal(out, expected)
