"""Грамматика имени и фамилии. Части склоняются раздельно, разделители сохраняются."""
from dataclasses import dataclass, replace
from functools import lru_cache
import re
import unicodedata

from .recognizer import PARTICLES

LETTERS = re.compile(r"[А-ЯЁа-яё]+")
PERSON_TEXT = re.compile(r"[А-ЯЁа-яё]+(?:[-‐‑ \t]+[А-ЯЁа-яё]+)*\Z")
CASES = {"nomn", "gent", "datv", "accs", "ablt", "loct"}
PREP = {"для":"gent","у":"gent","от":"gent","после":"gent","до":"gent","без":"gent",
        "из":"gent","к":"datv","ко":"datv","о":"loct","об":"loct","обо":"loct",
        "с":"ablt","со":"ablt","перед":"ablt","про":"accs","через":"accs"}
GENITIVE_HEADS = {"документы","документ","дело","карточка","карточке","карточку",
                  "обращению","заявление","заявления","анкета","справка","номер"}
DATIVE_VERBS = {"дать", "давать", "передать", "передавать", "вручить", "вручать",
                "отправить", "отправлять", "сообщить", "сообщать", "показать", "показывать",
                "выдать", "выдавать", "вернуть", "возвращать"}
ROLES = {"сотрудник":"masc","сотрудница":"femn","заявитель":"masc","заявительница":"femn",
         "господин":"masc","госпожа":"femn","гражданин":"masc","гражданка":"femn",
         "сын":"masc","дочь":"femn","пользовательница":"femn"}

@dataclass(frozen=True)
class Part:
    start: int
    end: int
    raw: str
    lemma: str
    gender: str | None
    number: str
    case: str
    particle: bool = False
    inferred: bool = False

@dataclass(frozen=True)
class Profile:
    kind: str
    raw: str
    parts: tuple
    key: tuple

@lru_cache(maxsize=1)
def analyzer():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()

def fold(s):
    return unicodedata.normalize("NFC",s).casefold().replace("ё","е")

@lru_cache(maxsize=24000)
def parse(s,kind):
    marker = "Name" if kind == "NAME" else "Surn"
    return tuple(p for p in analyzer().parse(s) if marker in p.tag)

def gender(p):
    g = p.tag.gender
    return str(g) if g else ("ms-f" if "ms-f" in p.tag else None)

def number(p):
    return str(p.tag.number or "sing")

def case(p):
    c = str(p.tag.case or "nomn")
    return {"gen1":"gent","gen2":"gent","loc1":"loct","loc2":"loct","acc2":"accs"}.get(c,c)

def is_dative_name_pair(text, given, surname):
    """Проверить сочетание имени общего рода и фамилии в позиции получателя."""
    if not re.fullmatch(r"[ \t]{1,6}", text[given.end:surname.start]):
        return False
    first = text[given.start:given.end]
    if not LETTERS.fullmatch(first) or not first[0].isupper():
        return False
    if not any(p.is_known and gender(p) == "ms-f" and case(p) == "datv"
               for p in parse(fold(first), "NAME")):
        return False
    preceding = re.search(r"([А-ЯЁа-яё]+)[ \t]+$", text[max(0, given.start - 40):given.start])
    if not preceding:
        return False
    cue = fold(preceding[1])
    if PREP.get(cue) != "datv" and not any(
        p.tag.POS == "VERB" and p.normal_form in DATIVE_VERBS for p in analyzer().parse(cue)
    ):
        return False
    surname_parts = LETTERS.findall(text[surname.start:surname.end])
    return bool(surname_parts and any(
        case(p) == "datv" and number(p) == "sing"
        for p in parse(fold(surname_parts[-1]), "SURNAME")
    ))

def context(text, a, b):
    before = re.split(r"[.!?;\n,:]",text[max(0,a-95):a])[-1]
    around = LETTERS.findall(before.lower())
    prep = PREP.get(around[-1]) if around else None
    role = ROLES.get(around[-1]) if around else None
    after = re.split(r"[.!?;\n,:]",text[b:b+75],maxsplit=1)[0]
    past = None
    if not prep:
        for word in LETTERS.findall(after)[:3]:
            p = analyzer().parse(word)[0]
            if p.tag.POS == "VERB" and "past" in p.tag:
                past = gender(p)
                break
            # Не ищем глагол в другом подлежащем.
            if p.tag.POS not in ("ADVB", "PRCL"):
                break
    context_case = prep or ("nomn" if role or past in ("masc","femn") else None)
    if not context_case and around and around[-1] in GENITIVE_HEADS:
        context_case = "gent"
    return role or (past if past in ("masc","femn") else None), context_case

def peer_group(text, spans, s):
    neighbor = None
    for x in spans:
        if x.kind == s.kind or x.kind not in ("NAME","SURNAME"):
            continue
        gap = text[x.end:s.start] if x.end <= s.start else (text[s.end:x.start] if s.end <= x.start else "")
        if re.fullmatch(r"[ \t]{1,6}",gap):
            neighbor = x
            break
    if neighbor:
        return min(s.start,neighbor.start), max(s.end,neighbor.end), neighbor
    return s.start,s.end,None

def _choose(word, kind, g, c):
    pp = parse(fold(word),kind)
    if not pp:
        # У неизвестной формы не предполагаем признак общего рода,
        # который в словаре есть только у конечного множества имён.
        # Контекст имеет приоритет; обычный разбор слова — приближение.
        ordinary = [p for p in analyzer().parse(word)
                    if p.tag.POS in ("NOUN","ADJF") and p.tag.case is not None]
        p = max(ordinary,key=lambda p:(int(c is not None and case(p)==c),
                int(g is not None and gender(p)==g),p.score),default=None)
        guess_g = gender(p) if p and gender(p) in ("masc","femn") else None
        guess_c = case(p) if p and case(p) in CASES else "nomn"
        return (fold(p.normal_form) if p else fold(word), g or guess_g,
                number(p) if p else "sing", c or guess_c, True)
    p = max(pp,key=lambda p: (
        int(c is not None and case(p)==c),
        int(g is not None and gender(p) in (g,"ms-f")),
        int(c is None and g is None and len(word)>2 and
            word.lower().endswith(("ова","ева","ина","ына")) and
            kind=="SURNAME" and gender(p)=="femn" and case(p)=="nomn"),
        int("Fixd" not in p.tag),
        p.score,
    ))
    g0 = gender(p)
    if g0 == "ms-f" and kind=="NAME" and g:
        g0=g
    c0=case(p)
    return fold(p.normal_form),g0,number(p),c0 if c0 in CASES else (c or "nomn"),False

def profile(text, span, person_spans):
    raw = text[span.start:span.end]
    if not PERSON_TEXT.fullmatch(raw):
        raise ValueError("Имя или фамилия имеют неподдерживаемую форму")
    a,b,peer = peer_group(text,person_spans,span)
    g,c = context(text,a,b)
    if peer:
        given, surname = (span, peer) if span.kind == "NAME" else (peer, span)
        if c is None and is_dative_name_pair(text, given, surname):
            c = "datv"
        # Согласование с соседним однозначным именем/фамилией.
        forms = parse(fold(text[peer.start:peer.end]),peer.kind)
        all_g = {gender(p) for p in forms if number(p)=="sing" and (not c or case(p)==c)}
        all_g &= {"masc","femn"}
        if len(all_g)==1 and not g:
            g=next(iter(all_g))
        all_c={case(p) for p in forms if not g or gender(p) in (g,"ms-f")}
        if len(all_c)==1 and not c:
            c=next(iter(all_c))
    matches=list(LETTERS.finditer(raw))
    parts=[]
    for i,m in enumerate(matches):
        is_particle=(span.kind=="SURNAME" and i < len(matches)-1 and fold(m[0]) in PARTICLES)
        if is_particle:
            parts.append(Part(m.start(),m.end(),m[0],fold(m[0]),None,"sing","nomn",True))
        else:
            lemma,g0,n,c0,guessed=_choose(m[0],span.kind,g,c)
            parts.append(Part(m.start(),m.end(),m[0],lemma,g0,n,c0,inferred=guessed))
    return make_profile(span.kind,raw,tuple(parts))

def make_profile(kind,raw,parts):
    lemmas=tuple(p.lemma for p in parts)
    # Однозначные мужское и женское употребления омонимичного имени
    # могут обозначать разных людей. Семейные фамилии получают общий корень.
    person_gender=tuple(p.gender for p in parts) if kind=="NAME" else ()
    return Profile(kind,raw,parts,(kind,lemmas,person_gender))

def reconcile(profiles):
    """Если род омонима однозначно выяснился в одном месте, переносим его
    на повторения той же леммы без подсказки. Два явных разных рода не объединяем."""
    genders={}
    for f in profiles:
        if f.kind != "NAME":
            continue
        for p in f.parts:
            if p.gender in ("masc","femn"):
                genders.setdefault(p.lemma,set()).add(p.gender)
    result=[]
    for f in profiles:
        if f.kind == "NAME":
            parts=[]
            for p in f.parts:
                gs=genders.get(p.lemma,set())
                parts.append(replace(p,gender=next(iter(gs))) if p.gender=="ms-f" and len(gs)==1 else p)
            f=make_profile(f.kind,f.raw,tuple(parts))
        result.append(f)
    return result

@lru_cache(maxsize=1)
def lemma_bank():
    from faker import Faker
    fake=Faker("ru_RU")
    fake.seed_instance(625)
    names={"NAME":set(("саша","женя","валя","слава","шура","оля","юра","дима","лёня")),
           "SURNAME":set()}
    for _ in range(1000):
        names["NAME"].update((fake.first_name_male(),fake.first_name_female()))
        names["SURNAME"].update((fake.last_name_male(),fake.last_name_female()))
    result={}
    for k,values in names.items():
        unique={}
        for word in sorted({fold(x) for x in values if LETTERS.fullmatch(x)}):
            for p in parse(word,k):
                if case(p)=="nomn" and number(p)=="sing":
                    unique.setdefault(fold(p.normal_form),[])
                    if str(p.tag) not in {str(x.tag) for x in unique[fold(p.normal_form)]}:
                        unique[fold(p.normal_form)].append(p)
        result[k]=unique
    return result

@lru_cache(maxsize=320)
def forms(kind,g,n,c):
    valid={}
    for lemma,parses in lemma_bank()[kind].items():
        for p in parses:
            if g in ("masc","femn") and gender(p)!=g:
                continue
            if g=="ms-f" and kind=="NAME" and gender(p)!="ms-f":
                continue
            tags={n,c}
            if n=="sing" and g in ("masc","femn"):
                tags.add(g)
            f=p.inflect(tags)
            if f is None or case(f)!=c or number(f)!=n:
                continue
            if n=="sing" and g in ("masc","femn") and gender(f)!=g:
                continue
            valid[lemma]=f.word
            break
    return valid

def spelling(source,word):
    if source.isupper():
        return word.upper()
    if source.islower():
        return word.lower()
    return word.capitalize()

def render_name(profile,lemmas):
    chunks=[]
    at=0
    for p,lemma in zip(profile.parts,lemmas):
        chunks.append(profile.raw[at:p.start])
        chunks.append(p.raw if p.particle else spelling(p.raw,forms(profile.kind,p.gender,p.number,p.case)[lemma]))
        at=p.end
    chunks.append(profile.raw[at:])
    return "".join(chunks)
