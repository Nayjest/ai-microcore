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
    results = texts.search("test_collection", "kitty", 2)
    assert "cat" == results[0]
    assert 2 == len(results)
    results = texts.search("test_collection", "folder", 20)
    assert 4 == len(results) and "catalog" == results[0]


def test_get():
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
    results = texts.get("test_collection")
    assert "cat" == results[0]
    assert 4 == len(results)
    results = texts.get("test_collection", limit=1, offset=1)
    assert "dog" == results[0]
    texts.save_many(
        "test_collection",
        [
            ("1", {"field": "value_a"}),
            ("2", {"field": "value_b"}),
        ],
    )
    results = texts.get("test_collection", where={"field": "value_a"})
    assert "1" == results[0]
    results = texts.get("test_collection", where_document={"$contains": "ca"})
    assert ["cat", "catalog"] == results
    results = texts.get(
        "test_collection", where_document={"$contains": "ca"}, limit=1, offset=1
    )
    assert ["catalog"] == results


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
    texts.save_many(
        cid,
        [
            ("1", {}),
        ],
    )
    texts.save_many(
        cid,
        [
            ("1", {}),
        ],
    )
    assert 1 == len(texts.get_all(cid))

    texts.save_many(
        cid,
        [
            ("1", {}),
            ("1", {}),
        ],
    )

    assert 1 == len(texts.get_all(cid))

    env().config.EMBEDDING_DB_ALLOW_DUPLICATES = True

    texts.clear(cid)
    texts.save_many(
        cid,
        [
            ("1", {}),
            ("1", {}),
        ],
    )
    assert 2 == len(texts.get_all(cid))


def test_count():
    cid = "test_count"
    texts.clear(cid)
    texts.save_many(
        cid,
        [
            ("1", {}),
            ("1", {}),
        ],
    )
    assert 2 == texts.count(cid)


def test_delete():
    cid = "test_delete"
    texts.clear(cid)
    texts.save_many(
        cid,
        [
            ("1", {"field": "value_a"}),
            ("2", {"field": "value_a"}),
            ("3", {"field": "value_aaa"}),
            ("4", {}),
            ("5", {}),
        ],
    )

    assert 5 == texts.count(cid)
    texts.delete(cid, {"field": "value_a"})
    assert "3,4,5" == ",".join(sorted(texts.get_all(cid)))
    # https://github.com/chroma-core/chroma/issues/4275#issuecomment-2807605563
    assert "4" == texts.find_one(cid, "4")
    texts.delete(cid, texts.find_one(cid, "4").id)
    assert "3,5" == ",".join(sorted(texts.get_all(cid)))
    texts.save(cid, "6")
    texts.save(cid, "7")
    texts.save(cid, "8")
    texts.delete(cid, [texts.find_one(cid, "3").id, texts.find_one(cid, "7").id])
    assert "5,6,8" == ",".join(sorted(texts.get_all(cid)))


def test_find_all():
    cid = "test_find_all"
    texts.clear(cid)
    texts.save_many(
        cid,
        [
            ("1", {"field": "value_a"}),
            ("2", {"field": "value_a"}),
            ("3", {"field": "value_a"}),
            ("4", {}),
            ("5", {}),
        ],
    )
    assert 3 == len(texts.find_all(cid, "", {"field": "value_a"}))
