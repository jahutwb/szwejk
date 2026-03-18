"""Dependency-tree walking and subtree alignment inside sentence-safe regions."""

from __future__ import annotations

from dataclasses import dataclass

from szwejk.align.embedding_eval import SentenceEncoder


@dataclass(slots=True)
class Subtree:
    root_index: int
    token_indices: list[int]
    text: str
    lemmas: list[str]
    upos_profile: list[str]
    deprel_profile: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "root_index": self.root_index,
            "token_indices": self.token_indices,
            "text": self.text,
            "lemmas": self.lemmas,
            "upos_profile": self.upos_profile,
            "deprel_profile": self.deprel_profile,
        }


@dataclass(slots=True)
class TreeAlignment:
    source_root_index: int
    target_root_index: int
    score: float
    node_score: float
    source_span: list[int]
    target_span: list[int]
    source_text: str
    target_text: str
    token_pairs: list[dict[str, object]]
    phrase_candidates: list[dict[str, object]]
    dependency_candidates: list[dict[str, object]]
    child_alignments: list["TreeAlignment"]


def extract_subtrees(tokens: list[dict[str, object]]) -> list[Subtree]:
    word_tokens = [token for token in tokens if token["kind"] == "word"]
    if not word_tokens:
        return []
    children: dict[int, list[int]] = {}
    for token in word_tokens:
        head = token.get("head")
        if isinstance(head, int) and head > 0:
            children.setdefault(head, []).append(int(token["index"]))

    roots = [int(token["index"]) for token in word_tokens if token.get("head") == 0]
    if not roots:
        roots = [int(word_tokens[0]["index"])]

    subtree_roots = set(roots)
    for root in roots:
        for child in children.get(root, []):
            subtree_roots.add(child)

    by_index = {int(token["index"]): token for token in tokens}
    return [
        _build_subtree(root_index=root_index, by_index=by_index, children=children)
        for root_index in sorted(subtree_roots)
    ]


def align_tree_units(
    source_tokens: list[dict[str, object]],
    target_tokens: list[dict[str, object]],
    *,
    encoder: SentenceEncoder | None = None,
) -> dict[str, list[dict[str, object]]]:
    source_words = [token for token in source_tokens if token["kind"] == "word"]
    target_words = [token for token in target_tokens if token["kind"] == "word"]
    if not source_words or not target_words:
        return {
            "token_pairs": [],
            "phrase_candidates": [],
            "dependency_candidates": [],
            "subtree_candidates": [],
        }

    source_by_index = {int(token["index"]): token for token in source_words}
    target_by_index = {int(token["index"]): token for token in target_words}
    source_children = _build_children(source_words)
    target_children = _build_children(target_words)
    source_subtrees = _subtree_lookup(source_tokens)
    target_subtrees = _subtree_lookup(target_tokens)
    embedding_lookup = _subtree_embeddings(list(source_subtrees.values()), list(target_subtrees.values()), encoder)

    source_roots = _root_indices(source_words)
    target_roots = _root_indices(target_words)
    root_pairs = _align_children(
        source_roots,
        target_roots,
        source_by_index=source_by_index,
        target_by_index=target_by_index,
        source_subtrees=source_subtrees,
        target_subtrees=target_subtrees,
        embedding_lookup=embedding_lookup,
        min_score=0.4,
    )

    tree_alignments: list[TreeAlignment] = []
    for source_root, target_root, root_score, node_score in root_pairs:
        alignment = _walk_parallel_trees(
            source_root,
            target_root,
            source_by_index=source_by_index,
            target_by_index=target_by_index,
            source_children=source_children,
            target_children=target_children,
            source_subtrees=source_subtrees,
            target_subtrees=target_subtrees,
            embedding_lookup=embedding_lookup,
            inherited_score=root_score,
            inherited_node_score=node_score,
        )
        if alignment is not None:
            tree_alignments.append(alignment)

    subtree_candidates = [_alignment_to_subtree_dict(alignment) for alignment in tree_alignments]
    token_pairs: list[dict[str, object]] = []
    phrase_candidates: list[dict[str, object]] = []
    dependency_candidates: list[dict[str, object]] = []
    for alignment in tree_alignments:
        _flatten_alignment(
            alignment,
            token_pairs=token_pairs,
            phrase_candidates=phrase_candidates,
            dependency_candidates=dependency_candidates,
        )

    return {
        "token_pairs": _dedupe_token_pairs(token_pairs),
        "phrase_candidates": _dedupe_span_candidates(phrase_candidates),
        "dependency_candidates": _dedupe_span_candidates(dependency_candidates),
        "subtree_candidates": _dedupe_span_candidates(subtree_candidates),
    }


def align_subtrees(
    source_tokens: list[dict[str, object]],
    target_tokens: list[dict[str, object]],
    *,
    encoder: SentenceEncoder | None = None,
) -> list[dict[str, object]]:
    return align_tree_units(source_tokens, target_tokens, encoder=encoder)["subtree_candidates"]


def subtree_similarity(
    source: Subtree,
    target: Subtree,
    *,
    embedding_lookup: dict[tuple[str, str], float] | None = None,
) -> float:
    return subtree_similarity_signals(source, target, embedding_lookup=embedding_lookup)["total"]


def subtree_similarity_signals(
    source: Subtree,
    target: Subtree,
    *,
    embedding_lookup: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    lemma = _set_overlap(set(source.lemmas), set(target.lemmas))
    upos = _sequence_overlap(source.upos_profile, target.upos_profile)
    deprel = _sequence_overlap(source.deprel_profile, target.deprel_profile)
    length = min(len(source.token_indices), len(target.token_indices)) / max(len(source.token_indices), len(target.token_indices))
    chars = _char_overlap(source.text, target.text)
    embedding = 0.0 if embedding_lookup is None else embedding_lookup.get((source.text, target.text), 0.0)
    total = (0.24 * lemma) + (0.18 * upos) + (0.18 * deprel) + (0.15 * length) + (0.10 * chars) + (0.15 * embedding)
    return {
        "lemma": lemma,
        "upos": upos,
        "deprel": deprel,
        "length": length,
        "chars": chars,
        "embedding": embedding,
        "total": total,
    }


def node_similarity(
    source_token: dict[str, object],
    target_token: dict[str, object],
) -> float:
    return node_similarity_signals(source_token, target_token)["total"]


def node_similarity_signals(
    source_token: dict[str, object],
    target_token: dict[str, object],
) -> dict[str, float]:
    source_norm = str(source_token.get("normalized", ""))
    target_norm = str(target_token.get("normalized", ""))
    source_lemma = str(source_token.get("lemma", ""))
    target_lemma = str(target_token.get("lemma", ""))
    lemma = 1.0 if source_lemma and source_lemma == target_lemma else 0.0
    norm = 1.0 if source_norm and source_norm == target_norm else _char_overlap(source_norm, target_norm)
    upos = 1.0 if source_token.get("upos") and source_token.get("upos") == target_token.get("upos") else 0.0
    deprel = 1.0 if source_token.get("deprel") and source_token.get("deprel") == target_token.get("deprel") else 0.0
    lexical = max(lemma, norm)
    total = (0.44 * lexical) + (0.26 * upos) + (0.20 * deprel) + (0.10 * norm)
    return {
        "lemma": lemma,
        "norm": norm,
        "upos": upos,
        "deprel": deprel,
        "total": total,
    }


def _build_subtree(*, root_index: int, by_index: dict[int, dict[str, object]], children: dict[int, list[int]]) -> Subtree:
    indices = sorted(_collect_descendants(root_index, children))
    tokens = [by_index[index] for index in indices]
    return Subtree(
        root_index=root_index,
        token_indices=indices,
        text=" ".join(str(token["text"]) for token in tokens),
        lemmas=[str(token["lemma"]) for token in tokens if token["kind"] == "word"],
        upos_profile=[str(token["upos"]) for token in tokens if token.get("upos")],
        deprel_profile=[str(token["deprel"]) for token in tokens if token.get("deprel")],
    )


def _build_children(tokens: list[dict[str, object]]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for token in tokens:
        head = token.get("head")
        if isinstance(head, int) and head > 0:
            children.setdefault(head, []).append(int(token["index"]))
    return {index: sorted(values) for index, values in children.items()}


def _subtree_lookup(tokens: list[dict[str, object]]) -> dict[int, Subtree]:
    word_tokens = [token for token in tokens if token["kind"] == "word"]
    by_index = {int(token["index"]): token for token in word_tokens}
    children = _build_children(word_tokens)
    return {
        index: _build_subtree(root_index=index, by_index=by_index, children=children)
        for index in sorted(by_index)
    }


def _root_indices(tokens: list[dict[str, object]]) -> list[int]:
    roots = [int(token["index"]) for token in tokens if token.get("head") == 0]
    if roots:
        return roots
    return [int(tokens[0]["index"])]


def _align_children(
    source_children: list[int],
    target_children: list[int],
    *,
    source_by_index: dict[int, dict[str, object]],
    target_by_index: dict[int, dict[str, object]],
    source_subtrees: dict[int, Subtree],
    target_subtrees: dict[int, Subtree],
    embedding_lookup: dict[tuple[str, str], float] | None,
    min_score: float,
) -> list[tuple[int, int, float, float]]:
    pairs: list[tuple[int, int, float, float]] = []
    target_start = 0
    for source_child in source_children:
        best: tuple[int, int, float, float] | None = None
        for offset, target_child in enumerate(target_children[target_start:], start=target_start):
            source_token = source_by_index[source_child]
            target_token = target_by_index[target_child]
            node_signals = node_similarity_signals(source_token, target_token)
            subtree_signals = subtree_similarity_signals(
                source_subtrees[source_child],
                target_subtrees[target_child],
                embedding_lookup=embedding_lookup,
            )
            node_score = node_signals["total"]
            subtree_score = subtree_signals["total"]
            combined = (0.55 * subtree_score) + (0.45 * node_score)
            if combined < min_score:
                continue
            candidate = (source_child, target_child, combined, node_score)
            if best is None or candidate[2] > best[2]:
                best = candidate
        if best is None:
            continue
        pairs.append(best)
        target_start = target_children.index(best[1]) + 1
    return pairs


def _walk_parallel_trees(
    source_root: int,
    target_root: int,
    *,
    source_by_index: dict[int, dict[str, object]],
    target_by_index: dict[int, dict[str, object]],
    source_children: dict[int, list[int]],
    target_children: dict[int, list[int]],
    source_subtrees: dict[int, Subtree],
    target_subtrees: dict[int, Subtree],
    embedding_lookup: dict[tuple[str, str], float] | None,
    inherited_score: float | None = None,
    inherited_node_score: float | None = None,
) -> TreeAlignment | None:
    source_subtree = source_subtrees[source_root]
    target_subtree = target_subtrees[target_root]
    subtree_signals = subtree_similarity_signals(source_subtree, target_subtree, embedding_lookup=embedding_lookup)
    subtree_score = subtree_signals["total"]
    node_score = inherited_node_score
    node_signals = node_similarity_signals(source_by_index[source_root], target_by_index[target_root])
    if node_score is None:
        node_score = node_signals["total"]
    combined_score = inherited_score if inherited_score is not None else ((0.55 * subtree_score) + (0.45 * node_score))
    if combined_score < 0.4:
        return None

    child_pairs = _align_children(
        source_children.get(source_root, []),
        target_children.get(target_root, []),
        source_by_index=source_by_index,
        target_by_index=target_by_index,
        source_subtrees=source_subtrees,
        target_subtrees=target_subtrees,
        embedding_lookup=embedding_lookup,
        min_score=0.43,
    )
    child_alignments: list[TreeAlignment] = []
    for child_source, child_target, child_score, child_node_score in child_pairs:
        child_alignment = _walk_parallel_trees(
            child_source,
            child_target,
            source_by_index=source_by_index,
            target_by_index=target_by_index,
            source_children=source_children,
            target_children=target_children,
            source_subtrees=source_subtrees,
            target_subtrees=target_subtrees,
            embedding_lookup=embedding_lookup,
            inherited_score=child_score,
            inherited_node_score=child_node_score,
        )
        if child_alignment is not None:
            child_alignments.append(child_alignment)

    token_pairs: list[dict[str, object]] = []
    dependency_candidates: list[dict[str, object]] = []
    if node_score >= 0.46:
        token_pair = _token_pair_from_nodes(
            source_by_index[source_root],
            target_by_index[target_root],
            score=node_score,
            signals=node_signals,
        )
        token_pairs.append(token_pair)
        dependency_candidates.append(_dependency_candidate_from_alignment(source_subtree, target_subtree, combined_score, subtree_signals))

    phrase_candidates: list[dict[str, object]] = []
    if child_alignments or (node_score >= 0.7 and len(source_subtree.token_indices) > 1 and len(target_subtree.token_indices) > 1):
        phrase_candidates.append(_phrase_candidate_from_alignment(source_subtree, target_subtree, combined_score, subtree_signals))

    return TreeAlignment(
        source_root_index=source_root,
        target_root_index=target_root,
        score=round(combined_score, 6),
        node_score=round(node_score, 6),
        source_span=[source_subtree.token_indices[0], source_subtree.token_indices[-1]],
        target_span=[target_subtree.token_indices[0], target_subtree.token_indices[-1]],
        source_text=source_subtree.text,
        target_text=target_subtree.text,
        token_pairs=token_pairs,
        phrase_candidates=phrase_candidates,
        dependency_candidates=dependency_candidates,
        child_alignments=child_alignments,
    )


def _collect_descendants(root_index: int, children: dict[int, list[int]]) -> set[int]:
    indices = {root_index}
    for child in children.get(root_index, []):
        indices.update(_collect_descendants(child, children))
    return indices


def _alignment_to_subtree_dict(alignment: TreeAlignment) -> dict[str, object]:
    return {
        "source_root_index": alignment.source_root_index,
        "target_root_index": alignment.target_root_index,
        "source_span": alignment.source_span,
        "target_span": alignment.target_span,
        "source_text": alignment.source_text,
        "target_text": alignment.target_text,
        "score": alignment.score,
        "node_score": alignment.node_score,
        "child_alignment_count": len(alignment.child_alignments),
        "relation": "tree_walk",
        "signals": {
            "total": alignment.score,
            "node_total": alignment.node_score,
        },
    }


def _token_pair_from_nodes(
    source_token: dict[str, object],
    target_token: dict[str, object],
    *,
    score: float,
    signals: dict[str, float] | None = None,
) -> dict[str, object]:
    source_norm = str(source_token.get("normalized", ""))
    target_norm = str(target_token.get("normalized", ""))
    source_lemma = str(source_token.get("lemma", ""))
    target_lemma = str(target_token.get("lemma", ""))
    if source_norm and source_norm == target_norm:
        relation = "exact"
    elif source_lemma and source_lemma == target_lemma:
        relation = "lemma_like"
    elif source_token.get("upos") and source_token.get("upos") == target_token.get("upos"):
        relation = "upos_cognate"
    else:
        relation = "cognate"
    return {
        "source_token_index": source_token["index"],
        "target_token_index": target_token["index"],
        "source_text": source_token["text"],
        "target_text": target_token["text"],
        "source_lemma": source_token["lemma"],
        "target_lemma": target_token["lemma"],
        "relation": relation,
        "score": round(score, 6),
        "signals": {key: round(value, 6) for key, value in (signals or {}).items()},
        "source_upos": source_token.get("upos"),
        "target_upos": target_token.get("upos"),
        "source_deprel": source_token.get("deprel"),
        "target_deprel": target_token.get("deprel"),
    }


def _phrase_candidate_from_alignment(
    source_subtree: Subtree,
    target_subtree: Subtree,
    score: float,
    signals: dict[str, float],
) -> dict[str, object]:
    return {
        "source_span": [source_subtree.token_indices[0], source_subtree.token_indices[-1]],
        "target_span": [target_subtree.token_indices[0], target_subtree.token_indices[-1]],
        "token_count": min(len(source_subtree.token_indices), len(target_subtree.token_indices)),
        "score": round(score, 6),
        "relation": "tree_phrase",
        "signals": {key: round(value, 6) for key, value in signals.items()},
    }


def _dependency_candidate_from_alignment(
    source_subtree: Subtree,
    target_subtree: Subtree,
    score: float,
    signals: dict[str, float],
) -> dict[str, object]:
    return {
        "source_span": [source_subtree.token_indices[0], source_subtree.token_indices[-1]],
        "target_span": [target_subtree.token_indices[0], target_subtree.token_indices[-1]],
        "token_count": min(len(source_subtree.token_indices), len(target_subtree.token_indices)),
        "score": round(score, 6),
        "relation": "tree_dependency",
        "root_source_index": source_subtree.root_index,
        "root_target_index": target_subtree.root_index,
        "signals": {key: round(value, 6) for key, value in signals.items()},
    }


def _flatten_alignment(
    alignment: TreeAlignment,
    *,
    token_pairs: list[dict[str, object]],
    phrase_candidates: list[dict[str, object]],
    dependency_candidates: list[dict[str, object]],
) -> None:
    token_pairs.extend(alignment.token_pairs)
    phrase_candidates.extend(alignment.phrase_candidates)
    dependency_candidates.extend(alignment.dependency_candidates)
    for child in alignment.child_alignments:
        _flatten_alignment(
            child,
            token_pairs=token_pairs,
            phrase_candidates=phrase_candidates,
            dependency_candidates=dependency_candidates,
        )


def _dedupe_token_pairs(token_pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_source: dict[int, dict[str, object]] = {}
    for pair in token_pairs:
        source_index = int(pair["source_token_index"])
        current = best_by_source.get(source_index)
        if current is None or float(pair["score"]) > float(current["score"]):
            best_by_source[source_index] = pair
    return sorted(best_by_source.values(), key=lambda pair: int(pair["source_token_index"]))


def _dedupe_span_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[tuple[tuple[int, ...], tuple[int, ...], str], dict[str, object]] = {}
    for candidate in candidates:
        relation = str(candidate.get("relation", ""))
        source_span = tuple(int(value) for value in candidate["source_span"])
        target_span = tuple(int(value) for value in candidate["target_span"])
        key = (source_span, target_span, relation)
        current = best.get(key)
        if current is None or float(candidate["score"]) > float(current["score"]):
            best[key] = candidate
    return sorted(best.values(), key=lambda item: (item["source_span"][0], item["target_span"][0]))


def _set_overlap(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _sequence_overlap(left: list[str], right: list[str]) -> float:
    return _set_overlap(set(left), set(right))


def _char_overlap(left: str, right: str) -> float:
    left_set = set(left.lower())
    right_set = set(right.lower())
    return _set_overlap(left_set, right_set)


def _subtree_embeddings(
    source_subtrees: list[Subtree],
    target_subtrees: list[Subtree],
    encoder: SentenceEncoder | None,
) -> dict[tuple[str, str], float] | None:
    if encoder is None:
        return None
    texts = ["passage: " + subtree.text for subtree in source_subtrees + target_subtrees]
    vectors = encoder.encode(texts)
    source_vectors = {
        subtree.text: [float(value) for value in vector]
        for subtree, vector in zip(source_subtrees, vectors[: len(source_subtrees)], strict=True)
    }
    target_vectors = {
        subtree.text: [float(value) for value in vector]
        for subtree, vector in zip(target_subtrees, vectors[len(source_subtrees) :], strict=True)
    }
    scores: dict[tuple[str, str], float] = {}
    for source_text, source_vector in source_vectors.items():
        for target_text, target_vector in target_vectors.items():
            scores[(source_text, target_text)] = sum(
                left * right for left, right in zip(source_vector, target_vector, strict=True)
            )
    return scores
