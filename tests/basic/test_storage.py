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
    assert (mc.storage.path / filename).exists()
    mc.storage.clean("tests_tmp")
    assert not (mc.storage.path / filename).exists()


def test_storage_file_exists():
    mc.storage.clean("tests_tmp")
    mc.storage.write("tests_tmp/file.json", "")
    assert (mc.storage.path / "tests_tmp/file.json").exists()
    mc.storage.write("tests_tmp/file", "")
    assert (mc.storage.path / "tests_tmp/file.txt").exists()
    mc.storage.clean("tests_tmp")


def test_json():
    mc.storage.clean("tests_tmp")
    mc.storage.write_json('test', [123])
    assert mc.storage.read_json('test')[0] == 123
    mc.storage.clean("tests_tmp")
