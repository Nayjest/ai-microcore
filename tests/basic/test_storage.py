import microcore as mc


def test_storage_read_write():
    mc.storage.clean("tests_tmp")
    filename = mc.storage.write("tests_tmp/test_file", "test content")
    content = mc.storage.read(filename)
    assert content == "test content"
    mc.storage.clean("tests_tmp")


def test_storage_write_existing():
    mc.storage.clean("tests_tmp")
    filename = mc.storage.write("tests_tmp/test_b", "old content")
    filename2 = mc.storage.write("tests_tmp/test_b", "new content")
    assert mc.storage.read(filename) == "old content"
    assert mc.storage.read(filename2) == "new content"
    assert filename != filename2
    mc.storage.clean("tests_tmp")


def test_storage_clean():
    mc.storage.clean("tests_tmp")
    filename = mc.storage.write("tests_tmp/file", "")
    assert (mc.storage.storage_path / filename).exists()
    mc.storage.clean("tests_tmp")
    assert not (mc.storage.storage_path / filename).exists()
