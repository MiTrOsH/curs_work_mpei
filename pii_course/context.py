"""Cached contextual morphology and syntax for checking neural person tags."""
from functools import lru_cache


@lru_cache(maxsize=1)
def _models():
    from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser

    embeddings = NewsEmbedding()
    return Segmenter(), NewsMorphTagger(embeddings), NewsSyntaxParser(embeddings)


@lru_cache(maxsize=8)
def document(text):
    from natasha import Doc

    segmenter, morph_tagger, syntax_parser = _models()
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    return doc
