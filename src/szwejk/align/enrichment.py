"""Hierarchical sentence enrichment for smaller-unit candidate extraction."""

from __future__ import annotations

from collections import Counter

from szwejk.align.embedding_eval import SentenceEncoder
from szwejk.align.diagnostics import augment_sentence_alignment
from szwejk.align.morphosyntax import analyze_sentence, tokenize_text
from szwejk.align.subtree import align_tree_units
from szwejk.align.sentences import ChapterSentenceAlignment


def build_sentence_enrichment(
    *,
    source_text: str,
    target_text: str,
    source_language: str,
    target_language: str,
    analysis_mode: str = "heuristic",
    subtree_encoder: SentenceEncoder | None = None,
) -> dict[str, object]:
    source_analysis = analyze_sentence(source_text, language=source_language, mode=analysis_mode)
    target_analysis = analyze_sentence(target_text, language=target_language, mode=analysis_mode)
    source_tokens = source_analysis.tokens
    target_tokens = target_analysis.tokens
    if _has_dependency_structure(source_tokens) and _has_dependency_structure(target_tokens):
        tree_alignment = align_tree_units(source_tokens, target_tokens, encoder=subtree_encoder)
        token_pairs = tree_alignment["token_pairs"]
        token_pairs = _complete_with_embedding_fallback(
            source_tokens,
            target_tokens,
            token_pairs,
            encoder=subtree_encoder,
        )
        phrase_candidates = tree_alignment["phrase_candidates"] or _fallback_phrase_candidates(token_pairs)
        dependency_candidates = tree_alignment["dependency_candidates"]
        subtree_candidates = tree_alignment["subtree_candidates"]
    else:
        token_pairs = _legacy_align_tokens(source_tokens, target_tokens)
        token_pairs = _complete_with_embedding_fallback(
            source_tokens,
            target_tokens,
            token_pairs,
            encoder=subtree_encoder,
        )
        phrase_candidates = _fallback_phrase_candidates(token_pairs)
        dependency_candidates = []
        subtree_candidates = []
    phrase_candidates.extend(_promote_fallback_spans(token_pairs, level="phrase"))
    subtree_candidates.extend(_promote_fallback_spans(token_pairs, level="subtree"))
    phrase_candidates = _dedupe_span_candidates(phrase_candidates)
    subtree_candidates = _dedupe_span_candidates(subtree_candidates)

    matched_source = {pair["source_token_index"] for pair in token_pairs}
    matched_target = {pair["target_token_index"] for pair in token_pairs}

    return {
        "analysis_mode": analysis_mode if source_analysis.mode == target_analysis.mode == analysis_mode else "mixed",
        "source_analysis": source_analysis.to_dict(),
        "target_analysis": target_analysis.to_dict(),
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "token_pairs": token_pairs,
        "token_group_candidates": [],
        "phrase_candidates": phrase_candidates,
        "dependency_candidates": dependency_candidates,
        "subtree_candidates": subtree_candidates,
        "unmatched_source_tokens": [token["index"] for token in source_tokens if token["kind"] == "word" and token["index"] not in matched_source],
        "unmatched_target_tokens": [token["index"] for token in target_tokens if token["kind"] == "word" and token["index"] not in matched_target],
    }


def _has_dependency_structure(tokens: list[dict[str, object]]) -> bool:
    return any(isinstance(token.get("head"), int) for token in tokens if token["kind"] == "word")


def _legacy_align_tokens(source_tokens: list[dict[str, object]], target_tokens: list[dict[str, object]]) -> list[dict[str, object]]:
    source_words = [token for token in source_tokens if token["kind"] == "word"]
    target_words = [token for token in target_tokens if token["kind"] == "word"]
    pairs: list[dict[str, object]] = []
    target_start = 1
    for source in source_words:
        best_pair: dict[str, object] | None = None
        for target in target_words:
            target_index = int(target["index"])
            if target_index < target_start:
                continue
            score, relation = _legacy_score_token_pair(source, target)
            if score < 0.45:
                continue
            candidate = {
                "source_token_index": source["index"],
                "target_token_index": target["index"],
                "source_text": source["text"],
                "target_text": target["text"],
                "source_lemma": source["lemma"],
                "target_lemma": target["lemma"],
                "relation": relation,
                "score": round(score, 6),
                "signals": {"total": round(score, 6)},
                "source_upos": source.get("upos"),
                "target_upos": target.get("upos"),
                "source_deprel": source.get("deprel"),
                "target_deprel": target.get("deprel"),
            }
            if best_pair is None or candidate["score"] > best_pair["score"]:
                best_pair = candidate
        if best_pair is None:
            continue
        pairs.append(best_pair)
        target_start = int(best_pair["target_token_index"]) + 1
    return pairs


def _legacy_score_token_pair(source: dict[str, object], target: dict[str, object]) -> tuple[float, str]:
    source_norm = str(source["normalized"])
    target_norm = str(target["normalized"])
    source_lemma = str(source["lemma"])
    target_lemma = str(target["lemma"])
    source_upos = source.get("upos")
    target_upos = target.get("upos")

    if source_norm and source_norm == target_norm:
        return 1.0, "exact"
    if source_lemma and source_lemma == target_lemma:
        return 0.92, "lemma_like"

    overlap = _char_trigram_jaccard(source_norm, target_norm)
    if source_upos and target_upos and source_upos == target_upos and overlap >= 0.28:
        return min(overlap + 0.08, 0.9), "upos_cognate"
    if overlap >= 0.35:
        return overlap, "cognate"
    return overlap, "weak"


def _fallback_phrase_candidates(token_pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    spans = _group_consecutive_pairs(token_pairs)
    phrase_candidates: list[dict[str, object]] = []
    for span in spans:
        if len(span) < 2:
            continue
        phrase_candidates.append(
            {
                "source_span": [int(span[0]["source_token_index"]), int(span[-1]["source_token_index"])],
                "target_span": [int(span[0]["target_token_index"]), int(span[-1]["target_token_index"])],
                "token_count": len(span),
                "score": round(sum(float(pair["score"]) for pair in span) / len(span), 6),
                "relation": "parallel_span",
                "signals": {
                    "total": round(sum(float(pair["score"]) for pair in span) / len(span), 6),
                },
            }
        )
    return phrase_candidates



def _group_consecutive_pairs(token_pairs: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    if not token_pairs:
        return []
    spans: list[list[dict[str, object]]] = []
    current = [token_pairs[0]]
    for pair in token_pairs[1:]:
        previous = current[-1]
        if (
            int(pair["source_token_index"]) == int(previous["source_token_index"]) + 1
            and int(pair["target_token_index"]) == int(previous["target_token_index"]) + 1
        ):
            current.append(pair)
            continue
        spans.append(current)
        current = [pair]
    spans.append(current)
    return spans




def _complete_with_embedding_fallback(
    source_tokens: list[dict[str, object]],
    target_tokens: list[dict[str, object]],
    token_pairs: list[dict[str, object]],
    *,
    encoder: SentenceEncoder | None,
) -> list[dict[str, object]]:
    if encoder is None:
        return token_pairs
    fallback_relations = {"dictionary", "embedding_assisted", "manual_forced"}
    source_words = [token for token in source_tokens if token["kind"] == "word"]
    target_words = [token for token in target_tokens if token["kind"] == "word"]
    used_source = {int(pair["source_token_index"]) for pair in token_pairs}
    used_target = {int(pair["target_token_index"]) for pair in token_pairs}
    unmatched_source = [token for token in source_words if int(token["index"]) not in used_source]
    unmatched_target = [token for token in target_words if int(token["index"]) not in used_target]
    if not unmatched_source or not unmatched_target:
        return token_pairs

    source_texts = ["query: " + str(token["text"]) for token in unmatched_source]
    target_texts = ["passage: " + str(token["text"]) for token in unmatched_target]
    encoded = encoder.encode(source_texts + target_texts)
    source_vectors = encoded[: len(unmatched_source)]
    target_vectors = encoded[len(unmatched_source) :]

    augmented = list(token_pairs)
    target_cursor = 0
    for source_token, source_vector in zip(unmatched_source, source_vectors, strict=True):
        best_index: int | None = None
        best_score = 0.0
        for candidate_index in range(target_cursor, len(unmatched_target)):
            target_token = unmatched_target[candidate_index]
            if int(target_token["index"]) in used_target:
                continue
            embedding = _cosine_similarity(source_vector, target_vectors[candidate_index])
            upos_bonus = 0.08 if source_token.get("upos") and source_token.get("upos") == target_token.get("upos") else 0.0
            deprel_bonus = 0.04 if source_token.get("deprel") and source_token.get("deprel") == target_token.get("deprel") else 0.0
            char_bonus = 0.08 * _char_trigram_jaccard(str(source_token["normalized"]), str(target_token["normalized"]))
            score = embedding + upos_bonus + deprel_bonus + char_bonus
            if score > best_score:
                best_score = score
                best_index = candidate_index
        if best_index is None or best_score < 0.72:
            continue
        target_token = unmatched_target[best_index]
        pair = {
            "source_token_index": source_token["index"],
            "target_token_index": target_token["index"],
            "source_text": source_token["text"],
            "target_text": target_token["text"],
            "source_lemma": source_token["lemma"],
            "target_lemma": target_token["lemma"],
            "relation": "embedding_assisted",
            "score": round(min(best_score, 0.99), 6),
            "signals": {
                "embedding": round(embedding, 6),
                "upos_bonus": round(upos_bonus, 6),
                "deprel_bonus": round(deprel_bonus, 6),
                "char_bonus": round(char_bonus, 6),
                "total": round(min(best_score, 0.99), 6),
            },
            "source_upos": source_token.get("upos"),
            "target_upos": target_token.get("upos"),
            "source_deprel": source_token.get("deprel"),
            "target_deprel": target_token.get("deprel"),
        }
        augmented.append(pair)
        used_source.add(int(source_token["index"]))
        used_target.add(int(target_token["index"]))
        target_cursor = best_index + 1

    augmented.sort(key=lambda pair: (int(pair["source_token_index"]), int(pair["target_token_index"])))
    cleaned: list[dict[str, object]] = []
    seen_source: set[int] = set()
    seen_target: set[int] = set()
    for pair in augmented:
        source_index = int(pair["source_token_index"])
        target_index = int(pair["target_token_index"])
        relation = str(pair.get("relation", ""))
        if source_index in seen_source or target_index in seen_target:
            if relation in fallback_relations:
                continue
        cleaned.append(pair)
        seen_source.add(source_index)
        seen_target.add(target_index)
    return cleaned


def _promote_fallback_spans(token_pairs: list[dict[str, object]], *, level: str) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for span in _group_consecutive_pairs(token_pairs):
        relations = {str(pair.get("relation", "")) for pair in span}
        if len(span) < 2 or not relations & {"dictionary", "embedding_assisted", "manual_forced"}:
            continue
        candidate = {
            "source_span": [int(span[0]["source_token_index"]), int(span[-1]["source_token_index"])],
            "target_span": [int(span[0]["target_token_index"]), int(span[-1]["target_token_index"])],
            "token_count": len(span),
            "score": round(sum(float(pair["score"]) for pair in span) / len(span), 6),
            "relation": "embedding_assisted" if "embedding_assisted" in relations else ("dictionary" if "dictionary" in relations else "manual_forced"),
            "signals": {
                "total": round(sum(float(pair["score"]) for pair in span) / len(span), 6),
            },
        }
        if level == "subtree":
            candidate.update(
                {
                    "source_root_index": int(span[0]["source_token_index"]),
                    "target_root_index": int(span[0]["target_token_index"]),
                    "source_text": " ".join(str(pair["source_text"]) for pair in span),
                    "target_text": " ".join(str(pair["target_text"]) for pair in span),
                }
            )
        candidates.append(candidate)
    return candidates


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
    return sorted(best.values(), key=lambda item: (item["source_span"][0], item["target_span"][0], str(item.get("relation", ""))))


def _char_trigram_jaccard(left: str, right: str) -> float:
    left_grams = set(_char_ngrams(left))
    right_grams = set(_char_ngrams(right))
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    if len(text) < n:
        return [text] if text else []
    return [text[index : index + n] for index in range(0, len(text) - n + 1)]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def build_sentence_enrichment_bundle(
    *,
    chapter_alignments: list[ChapterSentenceAlignment],
    source_language: str,
    target_language: str,
    analysis_mode: str = "heuristic",
    subtree_encoder: SentenceEncoder | None = None,
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    for row in chapter_alignments:
        sentence_alignment = augment_sentence_alignment(row.sentence_alignment)
        safe_depth = "sentence" if sentence_alignment["confidence_label"] in {"high", "medium"} else "paragraph"
        if safe_depth == "sentence":
            enrichment = build_sentence_enrichment(
                source_text=row.sentence_alignment.source_text,
                target_text=row.sentence_alignment.target_text,
                source_language=source_language,
                target_language=target_language,
                analysis_mode=analysis_mode,
                subtree_encoder=subtree_encoder,
            )
            enrichment_status = "sentence_safe"
        else:
            enrichment = {
                "analysis_mode": analysis_mode,
                "source_analysis": {"mode": analysis_mode, "language": source_language, "tokens": tokenize_text(row.sentence_alignment.source_text, language=source_language, mode=analysis_mode)},
                "target_analysis": {"mode": analysis_mode, "language": target_language, "tokens": tokenize_text(row.sentence_alignment.target_text, language=target_language, mode=analysis_mode)},
                "source_tokens": tokenize_text(row.sentence_alignment.source_text, language=source_language, mode=analysis_mode),
                "target_tokens": tokenize_text(row.sentence_alignment.target_text, language=target_language, mode=analysis_mode),
                "token_pairs": [],
                "phrase_candidates": [],
                "dependency_candidates": [],
                "subtree_candidates": [],
                "unmatched_source_tokens": [],
                "unmatched_target_tokens": [],
            }
            enrichment_status = "blocked_by_safe_depth"

        item = {
            "source_paragraph_id": row.source_paragraph_id,
            "target_paragraph_id": row.target_paragraph_id,
            "source_paragraph_index": row.source_paragraph_index,
            "target_paragraph_index": row.target_paragraph_index,
            "safe_alignment_depth": safe_depth,
            "enrichment_status": enrichment_status,
            "sentence_alignment": sentence_alignment,
            **enrichment,
        }
        items.append(item)
        counts[enrichment_status] += 1

    return {
        "summary": {
            "total_items": len(items),
            "sentence_safe_items": counts["sentence_safe"],
            "blocked_items": counts["blocked_by_safe_depth"],
        },
        "items": items,
    }
