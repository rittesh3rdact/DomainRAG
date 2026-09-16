"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

Combines multiple ranked lists (e.g. dense vector search + BM25 lexical
search) into a single ranking without needing the two lists' scores to be
on comparable scales -- only rank position matters.
"""

RRF_K = 60  # standard constant from the original paper; dampens the impact
# of a document's exact rank so results outside the top few still contribute


def reciprocal_rank_fusion(ranked_lists: list[list[str]]) -> list[str]:
    """Fuse ranked lists of item ids into one ranking, best first.

    Each inner list is a ranking of ids, best (most relevant) first. Ids
    absent from a list simply don't contribute a score from it.
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, item_id in enumerate(ranked_ids):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    return sorted(scores, key=scores.get, reverse=True)
