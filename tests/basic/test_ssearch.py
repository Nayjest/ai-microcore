from microcore import texts, env


def test_save_load():
    texts.clear("test_collection")
    texts.save("test_collection", "test text", {"test": "test"})
    assert texts.search("test_collection", "test text") == ["test text"]


def test_similarity():
    texts.clear("test_collection")
    texts.save_many(
        "test_collection",
        [
            "cat",
            "dog",
            "catalog",
            "kit",
        ],
    )
    assert texts.search("test_collection", "kitty", 1)[0] == "cat"


def test_metadata():
    texts.clear("test_collection")
    texts.save_many(
        "test_collection",
        [
            "1",
            "2",
            ("3", {"id": 33}),
            "4",
        ],
    )
    assert texts.find_one("test_collection", "3").metadata["id"] == 33


def test_get_all():
    cid = "test_collection3"
    texts.clear(cid)
    for i in range(20):
        texts.save(cid, f"test text {i}", {"test": "test"})
    assert len(texts.get_all(cid)) == 20


def test_uuid():
    cid = "test_uuid"
    texts.clear(cid)
    texts.save_many(cid, [
        ("1", {}),
    ])
    texts.save_many(cid, [
        ("1", {}),
    ])
    assert 1 == len(texts.get_all(cid))

    texts.save_many(cid, [
        ("1", {}),
        ("1", {}),
    ])

    assert 1 == len(texts.get_all(cid))

    env().config.EMBEDDING_DB_ALLOW_DUPLICATES = True

    texts.clear(cid)
    texts.save_many(cid, [
        ("1", {}),
        ("1", {}),
    ])
    assert 2 == len(texts.get_all(cid))
