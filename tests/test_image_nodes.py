import base64
import io
import json

import pytest
import torch
from PIL import Image

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


def data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_load_image_inspire_reads_the_picture_from_its_data(pack):
    image, mask = pack.image_util.LoadImageInspire().load_image("#DATA", data_url(Image.new("RGB", (6, 4), RED)))
    assert image.shape == (1, 4, 6, 3) and image[0, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert mask.shape == (1, 64, 64) and not mask.any()


def test_load_image_inspire_gives_the_transparency_as_mask(pack):
    picture = Image.new("RGBA", (4, 2), (0, 0, 255, 255))
    picture.putpixel((1, 0), (0, 0, 255, 0))
    _, mask = pack.image_util.LoadImageInspire().load_image("#DATA", data_url(picture))
    assert mask.tolist() == [[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]


@pytest.fixture
def folder(tmp_path):
    Image.new("RGB", (8, 8), RED).save(tmp_path / "unit-2.png")
    Image.new("RGB", (8, 8), GREEN).save(tmp_path / "unit-10.png")
    Image.new("RGB", (16, 4), BLUE).save(tmp_path / "unit-1.png")
    (tmp_path / "unit-notes.txt").write_text("not a picture")
    return tmp_path


def colors(images):
    return [tuple(round(value * 255) for value in image.reshape(-1, 3)[0].tolist()) for image in images]


@pytest.mark.parametrize("method,expected", [
    ("Alphabetical (ASC)", [BLUE, GREEN, RED]),
    ("Alphabetical (DESC)", [RED, GREEN, BLUE]),
    ("Numerical (ASC)", [BLUE, RED, GREEN]),
    ("Numerical (DESC)", [GREEN, RED, BLUE]),
])
def test_load_image_list_from_dir_keeps_each_picture_as_it_is_in_the_order_asked_for(pack, folder, method, expected):
    images, masks, paths = pack.image_util.LoadImagesFromDirList().load_images(str(folder), sort_method=method)
    assert colors(images) == expected
    assert [image.shape[1:3] for image in images] == [(4, 16) if color == BLUE else (8, 8) for color in expected]
    names = {BLUE: "unit-1.png", RED: "unit-2.png", GREEN: "unit-10.png"}
    assert paths == [str(folder / names[color]) for color in expected]
    assert all(mask.shape == (64, 64) and not mask.any() for mask in masks)


def test_load_image_list_from_dir_starts_at_the_index_and_stops_at_the_cap(pack, folder):
    images, _, _ = pack.image_util.LoadImagesFromDirList().load_images(str(folder), 1, 1, sort_method="Numerical (ASC)")
    assert colors(images) == [RED]


def test_load_image_batch_from_dir_resizes_every_picture_to_the_first(pack, folder):
    image, mask, count, paths = pack.image_util.LoadImagesFromDirBatch().load_images(str(folder), sort_method="Numerical (DESC)")
    assert count == 3 and image.shape == (3, 8, 8, 3)
    assert colors(image) == [GREEN, RED, BLUE]
    assert mask.shape == (3, 64, 64)
    assert json.loads(paths) == [str(folder / name) for name in ("unit-10.png", "unit-2.png", "unit-1.png")]


def test_load_image_batch_from_dir_gives_the_transparency_of_each_picture_at_its_size(pack, tmp_path):
    Image.new("RGB", (4, 4), RED).save(tmp_path / "unit-1.png")
    clear = Image.new("RGBA", (4, 4), (0, 0, 255, 0))
    clear.save(tmp_path / "unit-2.png")
    image, mask, count, _ = pack.image_util.LoadImagesFromDirBatch().load_images(str(tmp_path), sort_method="Alphabetical (ASC)")
    assert count == 2 and mask.shape == (2, 4, 4)
    assert not mask[0].any() and bool((mask[1] == 1).all())


def test_load_image_batch_from_dir_gives_a_single_picture_with_its_own_size(pack, folder):
    image, mask, count, paths = pack.image_util.LoadImagesFromDirBatch().load_images(str(folder), 1, sort_method="Alphabetical (ASC)")
    assert count == 1 and image.shape == (1, 4, 16, 3)
    assert json.loads(paths) == [str(folder / "unit-1.png")]


def test_load_image_batch_from_dir_names_the_pictures_it_loaded_from_the_index_to_the_cap(pack, folder):
    _, _, count, paths = pack.image_util.LoadImagesFromDirBatch().load_images(str(folder), 2, 1, sort_method="Numerical (ASC)")
    assert count == 2
    assert json.loads(paths) == [str(folder / name) for name in ("unit-2.png", "unit-10.png")]


def test_the_directory_loaders_say_what_is_missing(pack, tmp_path):
    for node in (pack.image_util.LoadImagesFromDirBatch(), pack.image_util.LoadImagesFromDirList()):
        with pytest.raises(FileNotFoundError):
            node.load_images(str(tmp_path / "unit-missing"))
        with pytest.raises(FileNotFoundError):
            node.load_images(str(tmp_path))


def frames(count):
    return torch.arange(count, dtype=torch.float32).reshape(count, 1, 1, 1).expand(count, 2, 2, 3).clone()


def test_change_image_batch_size_repeats_the_last_picture_or_cuts_the_batch(pack):
    node = pack.image_util.ChangeImageBatchSize
    assert node.doit(frames(2), 4, "simple")[0][:, 0, 0, 0].tolist() == [0, 1, 1, 1]
    assert node.doit(frames(4), 2, "simple")[0][:, 0, 0, 0].tolist() == [0, 1]


def test_change_latent_batch_size_changes_the_samples_only(pack):
    latent = {"samples": frames(2).movedim(-1, 1), "unit": "kept"}
    out = pack.image_util.ChangeLatentBatchSize.doit(latent, 3, "simple")[0]
    assert out["samples"][:, 0, 0, 0].tolist() == [0, 1, 1] and out["unit"] == "kept"
    assert latent["samples"].shape[0] == 2


@pytest.mark.parametrize("count,split,expected", [
    (5, 2, [[0], [1], [2, 3, 4]]),
    (3, 3, [[0], [1], [2], ["empty"]]),
    (2, 4, [[0], [1], ["empty"], ["empty"], ["empty"]]),
])
def test_image_batch_splitter_gives_split_count_pictures_and_the_remainder(pack, count, split, expected):
    out = pack.image_util.ImageBatchSplitter().doit(frames(count), split)
    assert len(out) == split + 1
    for image, items in zip(out, expected):
        if items == ["empty"]:
            assert image.shape == (1, 64, 64, 3) and not image.any()
        else:
            assert image[:, 0, 0, 0].tolist() == items


def test_latent_batch_splitter_keeps_the_other_entries_of_the_latent(pack):
    latent = {"samples": frames(3).movedim(-1, 1), "unit": "kept"}
    out = pack.image_util.LatentBatchSplitter().doit(latent, 2)
    assert [item["samples"][:, 0, 0, 0].tolist() for item in out] == [[0], [1], [2]]
    assert all(item["unit"] == "kept" for item in out)
    padded = pack.image_util.LatentBatchSplitter().doit(latent, 4)
    assert len(padded) == 5 and padded[3]["samples"].shape == (1, 4, 8, 8) and not padded[4]["samples"].any()


def color_map():
    image = torch.zeros(1, 10, 10, 3)
    image[0, :6, :, 0] = 1.0
    image[0, 6:9, :, 1] = 1.0
    return image


def test_color_map_to_masks_gives_a_mask_for_each_color_by_its_area(pack):
    (masks,) = pack.image_util.ColorMapToMasks().doit(color_map(), max_count=5, min_pixels=5)
    assert masks.shape == (3, 10, 10)
    assert [int(mask.sum()) for mask in masks] == [60, 30, 10]
    assert bool(masks[0, :6].all()) and bool(masks[1, 6:9].all()) and bool(masks[2, 9].all())


def test_color_map_to_masks_leaves_out_small_areas_and_keeps_to_max_count(pack):
    node = pack.image_util.ColorMapToMasks()
    assert node.doit(color_map(), max_count=5, min_pixels=20)[0].shape == (2, 10, 10)
    assert [int(mask.sum()) for mask in node.doit(color_map(), max_count=1, min_pixels=5)[0]] == [60]


def test_select_nth_mask(pack):
    masks = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1).expand(3, 2, 2)
    assert pack.image_util.SelectNthMask().doit(masks, 1)[0].tolist() == [[[1.0, 1.0], [1.0, 1.0]]]


@pytest.mark.parametrize("color,expected", [("#FF0000", (0.0, 1.0, 1.0)), ("00FF00", (120.0, 1.0, 1.0)), ("#0000FF", (240.0, 1.0, 1.0)),
                                            ("#808080", (0.0, 0.0, 128 / 255))])
def test_rgb_hex_to_hsv(pack, color, expected):
    assert pack.util_nodes.RGB_HexToHSV().doit(color) == pytest.approx(expected)
