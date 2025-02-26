from microcore.json_parsing import extract_block, ExtractStrategy


def test_pairs():
    pairs = (
        ("<", ">"),
        ("<tag>", "</tag>"),
        ("|||", "|||"),
        (
            "<very-long-tag-probably-larger-than-src-text>",
            "</very-long-tag-probably-larger-than-src-text>",
        ),
        (".", "."),
    )
    outer = (
        ("", "?"),
        ("", " ?"),
        ("?", ""),
        ("????", "???"),
        ("", ""),
    )
    contents = (
        "",
        "val",
    )

    for begin, end in pairs:
        for l, r in outer:
            for strat in ExtractStrategy:
                for content in contents:
                    assert (
                        extract_block(
                            f"{l}{begin}{content}{end}{r}",
                            begin,
                            end,
                            include_wrapper=False,
                            strategy=strat,
                        )
                        == content
                    )
                    assert (
                        extract_block(
                            f"{l}{begin}{content}{end}{r}",
                            begin,
                            end,
                            include_wrapper=True,
                            strategy=strat,
                        )
                        == f"{begin}{content}{end}"
                    )


def test_first_last_outer():
    pairs = (("<", ">"), ("begin", "end"))
    for b, e in pairs:
        assert (
            extract_block(f"0{b}1{b}2{e}3{e}4", b, e, True, ExtractStrategy.FIRST)
            == f"{b}1{b}2{e}"
        )
        assert (
            extract_block(f"0{b}1{b}2{e}3{e}4", b, e, False, ExtractStrategy.FIRST)
            == f"1{b}2"
        )

        assert (
            extract_block(f"0{b}1{b}2{e}3{e}4", b, e, True, ExtractStrategy.LAST)
            == f"{b}2{e}3{e}"
        )
        assert (
            extract_block(f"0{b}1{b}2{e}3{e}4", b, e, False, ExtractStrategy.LAST)
            == f"2{e}3"
        )

        assert (
            extract_block(f"0{b}1{b}2{e}3{e}4", b, e, True, ExtractStrategy.OUTER)
            == f"{b}1{b}2{e}3{e}"
        )
        assert (
            extract_block(f"0{b}1{b}2{e}3{e}4", b, e, False, ExtractStrategy.OUTER)
            == f"1{b}2{e}3"
        )
    b = "|"
    e = "|"
    assert (
        extract_block(f"0{b}1{b}2{e}3{e}4", b, e, True, ExtractStrategy.FIRST)
        == f"{b}1{b}"
    )
    assert (
        extract_block(f"0{b}1{b}2{e}3{e}4", b, e, False, ExtractStrategy.FIRST) == f"1"
    )

    assert (
        extract_block(f"0{b}1{b}2{e}3{e}4", b, e, True, ExtractStrategy.LAST)
        == f"{e}3{e}"
    )
    assert (
        extract_block(f"0{b}1{b}2{e}3{e}4", b, e, False, ExtractStrategy.LAST) == f"3"
    )

    assert (
        extract_block(f"0{b}1{b}2{e}3{e}4", b, e, True, ExtractStrategy.OUTER)
        == f"{b}1{b}2{e}3{e}"
    )
    assert (
        extract_block(f"0{b}1{b}2{e}3{e}4", b, e, False, ExtractStrategy.OUTER)
        == f"1{b}2{e}3"
    )
