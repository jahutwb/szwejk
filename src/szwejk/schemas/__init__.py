"""Schema exports for source and canonical corpus artifacts."""

from .corpus import BookMetadata, CanonicalBook, CanonicalChapter, CanonicalParagraph, CanonicalSentence
from .source import SourceFragmentRef

__all__ = [
    "BookMetadata",
    "CanonicalBook",
    "CanonicalChapter",
    "CanonicalParagraph",
    "CanonicalSentence",
    "SourceFragmentRef",
]
