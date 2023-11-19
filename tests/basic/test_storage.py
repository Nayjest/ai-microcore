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
    assert mc.storage.read(filename) == "new content"
    assert filename2 == filename
    assert mc.storage.read(f"tests_tmp/test_b_1") == "old content"
    filename3 = mc.storage.write("tests_tmp/test_b", "content 3", rewrite_existing=False)
    assert mc.storage.read(filename) == "new content"
    assert mc.storage.read(filename3) == "content 3"
    assert filename != filename3
    mc.storage.clean("tests_tmp")


def test_storage_clean():
    mc.storage.clean("tests_tmp")
    filename = mc.storage.write("tests_tmp/file", "")
    assert (mc.storage.path / filename).exists()
    mc.storage.clean("tests_tmp")
    assert not (mc.storage.path / filename).exists()


def test_default_ext():
    mc.storage.clean("tests_tmp")
    mc.storage.write("tests_tmp/file", "")
    assert (mc.storage.path / "tests_tmp/file").exists()
    mc.config().STORAGE_DEFAULT_FILE_EXT = "txt"
    mc.storage.write("tests_tmp/file", "")
    assert (mc.storage.path / "tests_tmp/file").exists()
    mc.storage.clean("tests_tmp")


def test_storage_file_exists():
    mc.storage.clean("tests_tmp")
    mc.storage.write("tests_tmp/file.json", "")
    assert (mc.storage.path / "tests_tmp/file.json").exists()
    assert mc.storage.exists("tests_tmp/file.json")
    mc.storage.write("tests_tmp/file.txt", "")
    assert (mc.storage.path / "tests_tmp/file.txt").exists()
    assert mc.storage.exists("tests_tmp/file.txt")
    assert mc.storage.exists("tests_tmp")
    assert not mc.storage.exists("tests_tmp/not_existing_file.txt")
    mc.storage.clean("tests_tmp")


def test_json():
    mc.storage.clean("tests_tmp")
    mc.storage.write_json("test", [123])
    assert mc.storage.read_json("test")[0] == 123
    mc.storage.clean("tests_tmp")


def test_copy():
    mc.storage.clean("tests_tmp")
    mc.storage.write_json("tests_tmp/folder/test.a", ['a'])
    mc.storage.write_json("tests_tmp/folder/test.b", ['b'])
    mc.storage.write_json("tests_tmp/folder/test.c", ['c'])
    mc.storage.copy("tests_tmp/folder", "tests_tmp/folder2", ["*.b"])

    assert mc.storage.read_json("tests_tmp/folder2/test.a") == ['a']
    assert mc.storage.read_json("tests_tmp/folder2/test.c") == ['c']
    assert mc.storage.read_json("tests_tmp/folder2/test.b", "none") == "none"

    # Test non wildcard
    mc.storage.copy("tests_tmp/folder", "tests_tmp/folder3", ["test.b"])
    assert mc.storage.read_json("tests_tmp/folder3/test.c") == ['c']
    assert mc.storage.read_json("tests_tmp/folder3/test.b", "none") == "none"

    # Test no exceptions
    mc.storage.copy("tests_tmp/folder", "tests_tmp/folder3")
    assert mc.storage.read_json("tests_tmp/folder3/test.b", "none") == ['b']

    # Test copy file
    mc.storage.copy("tests_tmp/folder/test.a", "tests_tmp/folder/test.d")
    assert mc.storage.read_json("tests_tmp/folder/test.d", "none") == ['a']

    # Test copy file to folder
    # @todo fails
    # mc.storage.copy("tests_tmp/folder/test.b", "tests_tmp/folder2")
    # assert mc.storage.read_json("tests_tmp/folder2/test.d", "none") == ['b']

    # Test overwrite
    mc.storage.write_json("tests_tmp/o1/test.a", ['a_new'])
    mc.storage.write_json("tests_tmp/o2/test.a", ['a'])
    mc.storage.copy("tests_tmp/o1", "tests_tmp/o2")
    assert mc.storage.read_json("tests_tmp/o2/test.a") == ['a_new']

    mc.storage.clean("tests_tmp")
