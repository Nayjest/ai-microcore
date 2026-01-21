def test_cache_dir():
    from microcore.file_cache import cache_dir
    assert cache_dir() == "cache/"
    assert cache_dir("test") == "cache/test/"


def test_build_cache_name():
    from microcore.file_cache import build_cache_name
    name = build_cache_name("arg1", prefix="test", key="val")
    assert name.startswith("cache/test/")
    assert name.endswith(".pkl")
    assert len(name.split("/")[-1]) == 68  # 64 hex + .pkl


def test_cache():
    from microcore import storage
    from microcore.file_cache import (
        cache_hit,
        build_cache_name,
        load_cache,
        flush_cache,
        save_cache,
        delete_cache
    )
    orig = storage.custom_path
    try:
        storage.custom_path = "tests"
        name = build_cache_name("something", prefix="tests_subfolder")
        flush_cache("tests_subfolder")
        assert flush_cache("non_existing_prefix") is False
        assert flush_cache("tests_subfolder") is False
        assert cache_hit(name) is False
        data = {"a": 1, "b": 2}
        save_cache(name, data)
        assert cache_hit(name) is True
        loaded = load_cache(name)
        assert loaded == data
        assert delete_cache(name) is True
        assert delete_cache(name) is False
        assert cache_hit(name) is False
        assert flush_cache("tests_subfolder") == True
    finally:
        storage.custom_path = orig
