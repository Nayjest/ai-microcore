import microcore as mc
from microcore import SearchResult


def test_fit_vector_search_to_tokens():
    mc.texts.clear("test_collection")
    raw_items = [str(i) for i in range(10)]
    mc.texts.save_many("test_collection", raw_items)
    res = mc.texts.search("test_collection", "qwe", n_results=10)
    # Check all loaded
    assert sorted(res) == raw_items

    fres = res.fit_to_token_size(3)
    # check fit
    assert len(fres) == 3
    assert any(i in raw_items for i in fres)

    # check that distances of fitted elements are smallest
    smallest_dist = sorted(i.distance for i in res)[:3]
    fitted_dist = sorted(i.distance for i in fres)
    assert fitted_dist == smallest_dist

    assert fres[0].num_tokens() == 1


def test_fit_vector_search_to_tokens_min_docs():
    mc.texts.clear("test_collection")
    raw_items = [str(i) for i in range(10)]
    mc.texts.save_many("test_collection", raw_items)
    res = mc.texts.search("test_collection", "qwe", n_results=10).fit_to_token_size(
        3, 4
    )
    assert len(res) == 4
    res = mc.texts.search("test_collection", "qwe", n_results=10).fit_to_token_size(
        5, 3
    )
    assert len(res) == 5


def test_num_tokens():
    assert (
        SearchResult("apple pineapple orange").num_tokens(encoding="cl100k_base") >= 3
    )
    assert SearchResult("Hi").num_tokens(for_model="gpt-4") <= 2
