"""Распознавание реквизитов: кандидаты, проверки и приоритеты."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import re
from .core import Span, tokens
from . import validators as v

S = v.SEP
D = r"[0-9]"
CERT = v.BIRTH.pattern[:-2]
PAS = v.PASSPORT.pattern[:-2]

@dataclass(frozen=True)
class Candidate:
    span: Span
    priority: float

# Окружение используется для неоднозначных чисел (например, 10 цифр ИНН/паспорта).
CUES = {
    "PASSPORT": r"паспорт(?:а|у|ом)?(?:\s+рф)?|паспортн\w*\s+данн\w*",
    "INN": r"инн|налогов\w*\s+номер",
    "BANK_CARD": r"(?:банковск\w*\s+)?карт(?:а|ы|е|ой|у)|pan",
    "OGRNIP": r"огрнип",
    "KLADR": r"(?:код\s+)?кладр",
    "IP": r"ip(?:v[46])?(?:[\s-]*адрес)?|сетев\w*\s+адрес",
}
CUE_PATTERNS = {k: re.compile(r"\b(?:" + pattern +
    r")\b(?:\s+(?:серия|серии|номер|номера|значение|гражданина|рф)){0,3}[\s:№=«\"#–—-]*$", re.I)
    for k, pattern in CUES.items()}

def has_cue(kind, text, start):
    p = CUE_PATTERNS.get(kind)
    return bool(p and p.search(text[max(0, start-80):start].split("\n")[-1]))

def boundary(pattern):
    return re.compile(r"(?<![\w])(?:" + pattern + r")(?![\w])")

# 15-значная карта в записи 4–6–5 и 19-значная в записи 4–4–4–4–3
# не должны разбиваться на отдельные реквизиты.
CARD_GROUP = rf"(?:{D}{{4}}{S}{D}{{6}}{S}{D}{{5}}|{D}{{4}}(?:{S}{D}{{4}}){{3}}(?:{S}{D}{{1,3}})?|{D}{{4}}(?:{S}{D}{{4}}){{2}}{S}{D}{{1,3}})"
KLADR_HEAD = rf"{D}{{2}}(?:{S}{D}{{3}}){{3}}"
KLADR_TAIL = rf"(?:(?:{S}{D}{{4}}){{2,3}}|(?:{S}{D}{{4,6}})?{S}{D}{{2}})"
PATTERNS = {
    "INN": boundary(rf"{D}{{4}}(?:{S}?{D}{{8}}|{S}?{D}{{6}})"),
    "PASSPORT": boundary(PAS),
    "BANK_CARD": boundary(rf"(?:{D}{{13,19}}|{CARD_GROUP})"),
    "OGRNIP": boundary(rf"(?:{D}{{15}}|3{S}?{D}{{2}}{S}{D}{{2}}{S}{D}{{9}}{S}{D})"),
    "KLADR": boundary(rf"(?:{D}{{23}}|{D}{{19}}|{D}{{17}}|{D}{{13}}|{KLADR_HEAD}{KLADR_TAIL})"),
    "BIRTH_CERT": boundary(CERT),
    "IP": re.compile(
        r"(?<![\w.:])(?:[0-9]{1,3}(?:\.[0-9]{1,3}){3})(?![\w:%]|\.[0-9])"),
    "EMAIL": re.compile(r"(?<![\w.+@-])" + v.EMAIL.pattern + r"(?![\w@-]|\.[\w-])"),
}
V6 = re.compile(r"(?<![\w:%])(?:[0-9a-fA-F]{0,4}:){1,7}[0-9A-Fa-f:.]{1,23}(?![\w:%])")
BASE = {"EMAIL": 240, "BIRTH_CERT": 210, "IP": 200, "OGRNIP": 145, "INN": 130,
        "PASSPORT": 128, "BANK_CARD": 125, "KLADR": 100}
NUMERIC_TYPES = ("INN", "PASSPORT", "BANK_CARD", "OGRNIP", "KLADR")

def rule_candidates(text):
    for kind, regexp in PATTERNS.items():
        for m in regexp.finditer(text):
            value = m[0]
            if not v.VALIDATORS[kind](value):
                continue
            cue = has_cue(kind, text, m.start())
            if kind in NUMERIC_TYPES and not cue and any(
                    has_cue(k, text, m.start()) for k in NUMERIC_TYPES if k != kind):
                # Реквизит с явной подписью не превращается в другой тип
                # только потому, что случайно проходит чужую проверку.
                continue
            compact = value == v.digits(value)
            if kind == "PASSPORT" and compact and not cue:
                continue
            if kind == "KLADR" and not cue:
                # Полный цифровой код без контекста может быть номером
                # карты или внутренним идентификатором.
                continue
            priority = BASE[kind] + (95 if cue else 0)
            if kind in ("PASSPORT", "BANK_CARD", "OGRNIP") and not compact:
                priority += 14
            yield Candidate(Span(m.start(), m.end(), kind), priority)
    for m in V6.finditer(text):
        value = m[0].rstrip(".")
        if value.count(":") < 2 or (value == "::" and not has_cue("IP", text, m.start())):
            continue
        if v.valid_ip(value):
            yield Candidate(Span(m.start(), m.start()+len(value), "IP"), BASE["IP"]+1)

def resolve(text, candidates):
    """Победитель определяется приоритетом, затем длиной, началом и типом."""
    selected = []
    lexemes = tokens(text)
    occupied = set()
    token_starts = [a for _, a, _ in lexemes]
    token_ends = [b for _, _, b in lexemes]
    for c in sorted(candidates, key=lambda c: (-c.priority, -(c.span.end-c.span.start),
                                                c.span.start, c.span.kind)):
        s = c.span
        l = bisect_right(token_ends, s.start)
        r = bisect_left(token_starts, s.end)
        ids = range(l, r)
        if any(j in occupied for j in ids) or any(s.start < x.end and x.start < s.end for x in selected):
            continue
        occupied.update(ids)
        selected.append(s)
    return sorted(selected)
