from app.rag.fusion import reciprocal_rank_fusion


def test_fuses_agreeing_lists_to_top():
    vector_ranking = ["a", "b", "c"]
    bm25_ranking = ["a", "c", "b"]

    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])

    assert fused[0] == "a"  # ranked #1 in both lists
    assert set(fused) == {"a", "b", "c"}


def test_item_present_in_only_one_list_still_included():
    fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert set(fused) == {"a", "b", "c"}


def test_empty_lists_return_empty():
    assert reciprocal_rank_fusion([[], []]) == []
