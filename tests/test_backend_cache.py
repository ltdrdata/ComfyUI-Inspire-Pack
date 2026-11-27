import pytest


def test_a_tag_keeps_as_many_entries_as_its_setting_and_drops_the_least_recently_used(pack):
    cache = pack.utils.TaggedCache({"ckpt": 2})
    for key in ("a", "b"):
        cache[key] = ("ckpt", key)
    assert cache["a"] == ("ckpt", "a")
    cache["c"] = ("ckpt", "c")
    assert "b" not in cache and "a" in cache and "c" in cache


@pytest.mark.parametrize("tag,size", [("unit ckpt shared", 5), ("latent", 100), ("image", 100), ("unit other", 20)])
def test_a_tag_without_a_setting_gets_the_default_size_of_its_kind(pack, tag, size):
    cache = pack.utils.TaggedCache({})
    for index in range(size + 1):
        cache[index] = (tag, index)
    assert [key for key, _ in cache.items()] == list(range(1, size + 1))


def test_a_key_stored_again_under_another_tag_moves_to_it(pack):
    cache = pack.utils.TaggedCache({})
    cache["k"] = ("first", 1)
    cache["k"] = ("second", 2)
    assert list(cache.items()) == [("k", ("second", 2))]
    del cache["k"]
    assert "k" not in cache and cache.get("k", "none") == "none"
    with pytest.raises(KeyError):
        cache["k"]


def test_cached_data_comes_back_from_retrieve(pack, run):
    nodes = pack.backend_support
    assert run(nodes.CacheBackendData, key=["unit key"], tag=["latent"], data=["L"]) == [["L"]]
    assert run(nodes.RetrieveBackendData, key=["unit key"]) == [["L"]]
    assert run(nodes.CacheBackendDataNumberKey, key=[7], tag=["image"], data=["I"]) == [["I"]]
    assert run(nodes.RetrieveBackendDataNumberKey, key=[7]) == [["I"]]
    assert run(nodes.RetrieveBackendData, key=["unit missing"]) == [[None]]


def test_a_cached_list_comes_back_as_the_same_list(pack, run):
    nodes = pack.backend_support
    assert run(nodes.CacheBackendDataList, key=["unit list"], tag=["image"], data=["a", "b", "c"]) == [["a", "b", "c"]]
    assert run(nodes.RetrieveBackendData, key=["unit list"]) == [["a", "b", "c"]]
    run(nodes.CacheBackendDataNumberKeyList, key=[3], tag=["image"], data=["x", "y"])
    assert run(nodes.RetrieveBackendDataNumberKey, key=[3]) == [["x", "y"]]


def test_the_reserved_key_is_not_stored(pack, run):
    nodes = pack.backend_support
    assert run(nodes.CacheBackendData, key=["*"], tag=["tag"], data=["data"]) == [[None]]
    assert list(nodes.cache.items()) == []


def test_retrieve_runs_again_when_its_key_is_stored_again(pack):
    nodes = pack.backend_support
    nodes.CacheBackendData.doit("unit key", "", 1)
    first = nodes.RetrieveBackendData.IS_CHANGED("unit key")
    assert nodes.RetrieveBackendData.IS_CHANGED("unit key") == first
    nodes.CacheBackendData.doit("unit key", "", 2)
    assert nodes.RetrieveBackendData.IS_CHANGED("unit key") != first


def test_remove_drops_one_key_and_the_reserved_key_drops_all(pack):
    nodes = pack.backend_support
    for key in ("unit a", "unit b"):
        nodes.CacheBackendData.doit(key, "", key)
    nodes.CacheBackendDataNumberKey.doit(5, "", 5)
    assert nodes.RemoveBackendData.doit("unit a", "signal") == ("signal",)
    assert "unit a" not in nodes.cache and "unit b" in nodes.cache
    assert nodes.RemoveBackendDataNumberKey.doit(5) == (None,)
    assert 5 not in nodes.cache
    nodes.RemoveBackendData.doit("*")
    assert list(nodes.cache.items()) == []


def test_is_cached_tells_whether_a_key_is_stored_and_runs_again_when_that_changes(pack):
    nodes = pack.backend_support
    node = nodes.IsCached()
    assert node.doit("unit key", "4") == (False,)
    unchanged = nodes.IsCached.IS_CHANGED("unit key", "4")
    assert nodes.IsCached.IS_CHANGED("unit key", "4") == unchanged
    nodes.CacheBackendData.doit("unit key", "", 1)
    assert node.doit("unit key", "4") == (True,)
    stored = nodes.IsCached.IS_CHANGED("unit key", "4")
    assert stored != unchanged and nodes.IsCached.IS_CHANGED("unit key", "4") == stored


def test_show_cached_info_writes_the_list_into_its_widget(pack):
    nodes = pack.backend_support
    nodes.CacheBackendData.doit("unit key", "latent", 1)
    assert nodes.ShowCachedInfo().doit("", "", "8") == {}
    ((event, data),) = pack.prompt_server.sent
    assert (event, data["node_id"], data["widget_name"]) == ("inspire-node-feedback", "8", "cache_info")
    assert data["data"] == nodes.ShowCachedInfo.get_data() and "unit key: latent\n" in data["data"]


def test_the_shared_checkpoint_loader_gives_the_cached_checkpoint_without_loading_it(pack):
    nodes = pack.backend_support
    nodes.update_cache("unit model key", "ckpt", (False, ("MODEL", "CLIP", "VAE")))
    nodes.update_cache("unit.safetensors", "ckpt", (False, ("M2", "C2", "V2")))
    loader = nodes.CheckpointLoaderSimpleShared()
    assert loader.doit("unit-not-on-disk.safetensors", " unit model key ") == ("MODEL", "CLIP", "VAE", "unit model key")
    assert loader.doit("unit.safetensors", "") == ("M2", "C2", "V2", "unit.safetensors")
    assert loader.doit("unit-not-on-disk.safetensors", "unit model key", "Read Only") == ("MODEL", "CLIP", "VAE", "unit model key")
    with pytest.raises(Exception, match="key_opt cannot be omit"):
        loader.doit("unit.safetensors", "", "Read Only")


def test_the_shared_checkpoint_loader_runs_again_when_its_key_is_cached_again(pack):
    nodes = pack.backend_support
    changed = nodes.CheckpointLoaderSimpleShared.IS_CHANGED
    first = changed("unit.safetensors", "unit key")
    assert changed("unit.safetensors", "unit key") == first
    nodes.update_cache("unit key", "ckpt", (False, ("M", "C", "V")))
    nodes.update_cache("unit key", "ckpt", (False, ("M", "C", "V")))
    assert changed("unit.safetensors", "unit key") != first
