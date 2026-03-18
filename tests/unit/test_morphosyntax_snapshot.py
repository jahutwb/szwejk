from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from szwejk.align import morphosyntax


class MorphosyntaxSnapshotUnitTest(unittest.TestCase):
    def test_analyze_sentence_prefers_persisted_stanza_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            payload = {
                "language": "pl",
                "entries": [
                    {
                        "text": "Pan Ferdynand siedzi.",
                        "tokens": [
                            {
                                "index": 1,
                                "text": "Pan",
                                "kind": "word",
                                "normalized": "pan",
                                "lemma": "pan",
                                "lemma_source": "stanza",
                                "is_stopword": False,
                                "upos": "NOUN",
                                "feats": "Case=Nom",
                                "head": 2,
                                "deprel": "nsubj",
                                "analysis_source": "stanza",
                            }
                        ],
                    }
                ],
            }
            (cache_dir / "pl_comparable_snapshot.json").write_text(json.dumps(payload), encoding="utf-8")
            morphosyntax._analyze_sentence_cached.cache_clear()
            morphosyntax._load_persisted_stanza_snapshot.cache_clear()
            with patch.dict(os.environ, {"SZWJK_STANZA_CACHE_DIR": str(cache_dir)}, clear=False):
                analysis = morphosyntax.analyze_sentence("Pan Ferdynand siedzi.", language="pl", mode="stanza")

            self.assertEqual(analysis.mode, "stanza")
            self.assertEqual(analysis.tokens[0]["analysis_source"], "stanza")
            self.assertEqual(analysis.tokens[0]["upos"], "NOUN")
            morphosyntax._analyze_sentence_cached.cache_clear()
            morphosyntax._load_persisted_stanza_snapshot.cache_clear()


if __name__ == "__main__":
    unittest.main()
