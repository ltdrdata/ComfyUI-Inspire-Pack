import asyncio
from pathlib import Path

import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = "---- [TagCache Settings] ----\n"


def call(pack, *requests):
    async def run():
        app = web.Application()
        app.add_routes(pack.routes)
        async with TestClient(TestServer(app)) as client:
            answers = []
            for method, path, body in requests:
                response = await client.request(method, path, data=body)
                answers.append((response.status, await response.text()))
            return answers
    return asyncio.run(run())


def get(pack, path):
    return call(pack, ("GET", path, None))[0]


def test_the_pack_serves_these_routes(pack):
    assert sorted((route.method, route.path) for route in pack.routes) == [
        ("GET", "/inspire/cache/clear"), ("GET", "/inspire/cache/list"), ("GET", "/inspire/cache/remove"),
        ("GET", "/inspire/prompt_builder"), ("POST", "/inspire/cache/settings")]


def test_prompt_builder_gives_the_presets_of_a_category(pack):
    presets = yaml.safe_load((ROOT / "resources" / "prompt-builder.yaml").read_text(encoding="utf-8"))
    category = next(iter(presets))
    status, text = get(pack, f"/inspire/prompt_builder?category={category}")
    assert status == 200 and yaml.safe_load(text) == {"presets": presets[category]}
    assert get(pack, "/inspire/prompt_builder?category=unit-no-such-category")[1] == '{"presets": []}'
    assert get(pack, "/inspire/prompt_builder")[1] == '{"presets": []}'


def cached(pack):
    nodes = pack.backend_support
    nodes.CacheBackendData.doit("unit model", "ckpt", "M")
    nodes.CacheBackendData.doit("unit latent", "", "L")
    nodes.CacheBackendDataNumberKey.doit(7, "image", "I")


def test_cache_list_shows_each_key_with_its_tag_and_the_size_of_each_tag(pack):
    cached(pack)
    status, text = get(pack, "/inspire/cache/list")
    assert status == 200
    strings, numbers, settings = text.split("\n\n")
    assert strings == "---- [String Key Caches] ----\nunit model: ckpt\nunit latent: N/A(tag)"
    assert numbers == "---- [Number Key Caches] ----\n7: image"
    assert settings == SETTINGS + "ckpt: 5\nlatent: 100\nimage: 100\n: 20\n"


def test_cache_remove_drops_one_key_and_clear_drops_all(pack):
    cached(pack)
    assert get(pack, "/inspire/cache/remove?key=unit model")[0] == 200
    assert "unit model" not in pack.backend_support.cache and "unit latent" in pack.backend_support.cache
    assert get(pack, "/inspire/cache/clear")[0] == 200
    assert list(pack.backend_support.cache.items()) == []
    assert pack.backend_support.RetrieveBackendData.doit("unit latent") == ([None],)


def test_saved_cache_settings_change_how_many_entries_a_tag_keeps(pack):
    cached(pack)
    listed = get(pack, "/inspire/cache/list")[1]
    edited = listed.replace("ckpt: 5", "ckpt: 1")
    assert call(pack, ("POST", "/inspire/cache/settings", edited)) == [(200, "OK")]
    assert "ckpt: 1" in get(pack, "/inspire/cache/list")[1]
    pack.backend_support.CacheBackendData.doit("unit second model", "ckpt", "M2")
    cache = pack.backend_support.cache
    assert "unit model" not in cache and cache["unit second model"] == ("ckpt", (False, "M2"))
    assert cache["unit latent"] == ("", (False, "L")) and cache[7] == ("image", (False, "I"))


def test_settings_that_are_not_numbers_are_refused_and_the_cache_is_kept(pack):
    cached(pack)
    status, text = call(pack, ("POST", "/inspire/cache/settings", SETTINGS + "ckpt: many"))[0]
    assert status == 500 and "many" in text
    assert pack.backend_support.cache["unit model"] == ("ckpt", (False, "M"))
    assert "ckpt: 5" in get(pack, "/inspire/cache/list")[1]
