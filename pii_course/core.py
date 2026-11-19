"""Символьные интервалы и общая токенизация для обучения и оценки."""
from dataclasses import dataclass
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("SURNAME", "NAME", "INN", "PASSPORT", "BANK_CARD", "OGRNIP",
         "IP", "EMAIL", "BIRTH_CERT", "KLADR")
# Слово с дефисом — одна единица. Цифровые группы и пунктуация
# остаются отдельными единицами, так что BIO описывает и составные реквизиты.
LEXEMES = re.compile(r"[^\W\d_]+(?:[-‐‑][^\W\d_]+)*|[0-9]+|[^\s]", re.UNICODE)
WORD = re.compile(r"[А-ЯЁа-яё]+(?:[-‐‑][А-ЯЁа-яё]+)*\Z")

@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    kind: str

    def triple(self):
        return [self.kind, self.start, self.end]

def spans_from(triples, text):
    spans = sorted(Span(a, b, k) for k, a, b in triples)
    last = 0
    for s in spans:
        if (s.kind not in KINDS or not 0 <= s.start < s.end <= len(text)
                or s.start < last):
            raise ValueError(f"Неверная разметка или пересечение: {s}")
        last = s.end
    return spans

def tokens(text):
    return [(m[0], m.start(), m.end()) for m in LEXEMES.finditer(text)]

def bio(lexemes, spans):
    """Проекция непересекающихся интервалов на общие токены."""
    labels = ["O"] * len(lexemes)
    i = 0
    for s in sorted(spans):
        while i < len(lexemes) and lexemes[i][2] <= s.start:
            i += 1
        first = i
        while i < len(lexemes) and lexemes[i][1] < s.end:
            labels[i] = ("B-" if i == first else "I-") + s.kind
            i += 1
    return labels

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
