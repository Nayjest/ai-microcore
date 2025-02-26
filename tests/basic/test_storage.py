import microcore as mc
import pytest


def test_storage_read_write():
    mc.storage.delete("tests_tmp")
    filename = mc.storage.write("tests_tmp/test_file", "test content")
    content = mc.storage.read(filename)
    assert content == "test content"
    mc.storage.delete("tests_tmp")


def test_storage_write_existing():
    mc.storage.delete("tests_tmp")
    filename = mc.storage.write("tests_tmp/test_b", "old content")
    filename2 = mc.storage.write("tests_tmp/test_b", "new content")
    assert mc.storage.read(filename) == "new content"
    assert filename2 == filename
    assert mc.storage.read(f"tests_tmp/test_b_1") == "old content"
    filename3 = mc.storage.write(
        "tests_tmp/test_b", "content 3", rewrite_existing=False
    )
    assert mc.storage.read(filename) == "new content"
    assert mc.storage.read(filename3) == "content 3"
    assert filename != filename3
    mc.storage.delete("tests_tmp")


def test_list_files():
    mc.storage.delete("tmp")
    mc.storage.write("tmp/file_1", "content")
    mc.storage.write("tmp/file_2", "content")
    lf = mc.storage.list_files
    assert set([str(f) for f in lf("tmp")]) == {"file_1", "file_2"}
    assert set(
        [str(f) for f in lf("tmp", relative_to=mc.storage.path, posix=True)]
    ) == {"tmp/file_1", "tmp/file_2"}
    assert [str(f) for f in lf("tmp", exclude=["file_1"])] == ["file_2"]
    assert [str(f) for f in lf("tmp", exclude=["*_1"])] == ["file_2"]
    assert [str(f) for f in lf("tmp", exclude=["file_*"])] == []
    assert [
        str(f)
        for f in lf("tmp", exclude=["file_1"], relative_to=mc.storage.path, posix=True)
    ] == ["tmp/file_2"]
    assert [
        str(f)
        for f in lf("tmp", exclude=["*_1"], relative_to=mc.storage.path, posix=True)
    ] == ["tmp/file_2"]
    assert [
        str(f) for f in lf("tmp", exclude=["file_*"], relative_to=mc.storage.path)
    ] == []
    mc.storage.delete("tmp")


def test_storage_delete():
    mc.storage.delete("tests_tmp")

    # Delete folder
    filename = mc.storage.write("tests_tmp/file", "")
    assert (mc.storage.path / filename).exists()
    mc.storage.delete("tests_tmp")
    assert not (mc.storage.path / filename).exists()

    # Delete file
    filename = mc.storage.write("tests_tmp/file", "")
    mc.storage.delete(filename)
    assert not (mc.storage.path / filename).exists()


def test_default_ext():
    mc.storage.delete("tests_tmp")
    mc.storage.write("tests_tmp/file", "")
    assert (mc.storage.path / "tests_tmp/file").exists()
    mc.config().STORAGE_DEFAULT_FILE_EXT = "txt"
    mc.storage.write("tests_tmp/file", "")
    assert (mc.storage.path / "tests_tmp/file").exists()
    mc.storage.delete("tests_tmp")


def test_storage_file_exists():
    mc.storage.delete("tests_tmp")
    mc.storage.write("tests_tmp/file.json", "")
    assert (mc.storage.path / "tests_tmp/file.json").exists()
    assert mc.storage.exists("tests_tmp/file.json")
    mc.storage.write("tests_tmp/file.txt", "")
    assert (mc.storage.path / "tests_tmp/file.txt").exists()
    assert mc.storage.exists("tests_tmp/file.txt")
    assert mc.storage.exists("tests_tmp")
    assert not mc.storage.exists("tests_tmp/not_existing_file.txt")
    mc.storage.delete("tests_tmp")


def test_json():
    mc.storage.delete("tests_tmp")
    mc.storage.write_json("test", [123])
    assert mc.storage.read_json("test")[0] == 123
    mc.storage.delete("tests_tmp")


def test_copy():
    mc.storage.delete("tests_tmp")
    mc.storage.write_json("tests_tmp/folder/test.a", ["a"])
    mc.storage.write_json("tests_tmp/folder/test.b", ["b"])
    mc.storage.write_json("tests_tmp/folder/test.c", ["c"])
    mc.storage.copy("tests_tmp/folder", "tests_tmp/folder2", ["*.b"])

    assert mc.storage.read_json("tests_tmp/folder2/test.a") == ["a"]
    assert mc.storage.read_json("tests_tmp/folder2/test.c") == ["c"]
    assert mc.storage.read_json("tests_tmp/folder2/test.b", "none") == "none"

    # Test non wildcard
    mc.storage.copy("tests_tmp/folder", "tests_tmp/folder3", ["test.b"])
    assert mc.storage.read_json("tests_tmp/folder3/test.c") == ["c"]
    assert mc.storage.read_json("tests_tmp/folder3/test.b", "none") == "none"

    # Test no exceptions
    mc.storage.copy("tests_tmp/folder", "tests_tmp/folder3")
    assert mc.storage.read_json("tests_tmp/folder3/test.b", "none") == ["b"]

    # Test copy file
    mc.storage.copy("tests_tmp/folder/test.a", "tests_tmp/folder/test.d")
    assert mc.storage.read_json("tests_tmp/folder/test.d", "none") == ["a"]

    # Test copy file to folder
    mc.storage.write("tests_tmp/copy_to_folder.txt", "ok")
    mc.storage.copy("tests_tmp/copy_to_folder.txt", "tests_tmp/folder2")
    assert mc.storage.read("tests_tmp/folder2/copy_to_folder.txt") == "ok"

    # Test overwrite
    mc.storage.write_json("tests_tmp/o1/test.a", ["a_new"])
    mc.storage.write_json("tests_tmp/o2/test.a", ["a"])
    mc.storage.copy("tests_tmp/o1", "tests_tmp/o2")
    assert mc.storage.read_json("tests_tmp/o2/test.a") == ["a_new"]

    mc.storage.delete("tests_tmp")


def test_create_no_name():
    fn = mc.storage.write("test_data")
    assert (mc.storage.path / fn).exists()
    assert mc.storage.read(fn) == "test_data"
    mc.storage.delete(fn)


def test_read_default():
    assert mc.storage.read("non_existing_file", default="default") == "default"
    with pytest.raises(FileNotFoundError):
        mc.storage.read_json("non_existing_file")


def test_read_json_default():
    assert mc.storage.read_json("non_existing_file", default={"default": "value"}) == {
        "default": "value"
    }
    with pytest.raises(FileNotFoundError):
        mc.storage.read_json("non_existing_file")
