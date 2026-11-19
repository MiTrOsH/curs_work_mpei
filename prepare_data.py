"""Сборка зафиксированных выборок FactRuEval и синтетических документов."""
import argparse
from collections import Counter
from hashlib import sha256
import ipaddress
from pathlib import Path
import random
import re
from zipfile import ZipFile

from faker import Faker
import pymorphy3
from pii_course.core import KINDS, ROOT, Span, spans_from, tokens, write_json, write_jsonl
from pii_course import validators as v

CORPUS_REVISION = "dfe96237fce79fde13b2021d1938587e4ea3042e"
STRUCTURED = KINDS[2:]
REGIONS = ("16", "23", "50", "52", "63", "66", "74", "77", "78")
INTRO = {
    "train": (
        "В отдел обратился {person}. Данные заявления проверены.",
        "Сотрудник связался с {person}. Ниже указаны реквизиты для учёта.",
        "Для {person} открыт новый черновик. Сведения внесены в систему.",
        "Карточку заполнил{a} {person}. Данные сохранены во внутренней форме.",
        "{person} передал{a} документы в канцелярию. Реквизиты переписаны.",
        "Дело {person} принято на проверку. Указаны следующие сведения.",
    ),
    "validation": (
        "По обращению {person} создана запись. Уточнённые поля перечислены далее.",
        "Выписку направили к {person}. Карточка содержит новые реквизиты.",
        "{person} согласовал{a} изменения в анкете. Сохранились такие поля.",
    ),
    "test": (
        "Архивная справка подготовлена для {person}. При сверке просмотрены следующие поля:",
        "В протоколе указано, что {person} подтвердил{a} сведения перед регистрацией.",
        "В отделе сообщили о {person}. Для проверки по заявлению доступны реквизиты:",
        "{person} подписал{a} согласие. В приложение к заявлению внесли такие данные:",
        "Пакет от {person} принят без замечаний. В него включены следующие сведения:",
        "В карточке {person} уточнены данные. Результат проверки записали ниже.",
        "Архивариус встретился с {person}. В опись добавлены следующие реквизиты:",
        "{person} подал{a} запрос на исправление записей в журнале.",
    ),
}
CASES = {"train": ("nomn", "ablt", "gent", "nomn", "nomn", "gent"),
         "validation": ("gent", "datv", "nomn"),
         "test": ("gent", "nomn", "loct", "nomn", "gent", "gent", "ablt", "nomn")}
FIELDS = {
    "INN": ("ИНН", "налоговый номер", "ИНН"),
    "PASSPORT": ("паспорт", "паспортные данные", "паспорт гражданина"),
    "BANK_CARD": ("карта", "номер банковской карты", "PAN"),
    "OGRNIP": ("ОГРНИП", "ОГРНИП", "ОГРНИП"),
    "IP": ("IP-адрес", "сетевой адрес", "IP"),
    "EMAIL": ("E-mail", "электронная почта", "адрес для уведомлений"),
    "BIRTH_CERT": ("свидетельство о рождении", "номер свидетельства о рождении", "бланк свидетельства о рождении"),
    "KLADR": ("КЛАДР", "код КЛАДР", "КЛАДР"),
}

def annotation(path, raw_text, raw_spans, source):
    text = raw_text.decode("utf-8-sig")
    parsed = []
    for line in raw_spans.decode("utf-8-sig").splitlines():
        fields = line.partition("#")[0].split()
        if len(fields) < 6 or fields[1] not in ("name", "surname"):
            continue
        a, length = map(int, fields[2:4])
        b = a + length
        if not 0 <= a < b <= len(text):
            raise ValueError(f"Неверные смещения {path}: {line}")
        while a < b and text[a].isspace():
            a += 1
        while a < b and text[b-1].isspace():
            b -= 1
        if a < b:
            parsed.append(Span(a, b, fields[1].upper()))
    # Сначала объединяем перекрытия одного типа. При разном типе
    # выбираем более короткий атрибут; остатки длинного не теряем.
    merged = []
    for s in sorted(set(parsed), key=lambda x: (x.kind, x.start, x.end)):
        if merged and merged[-1].kind == s.kind and s.start < merged[-1].end:
            old = merged.pop()
            s = Span(old.start, max(s.end, old.end), s.kind)
        merged.append(s)
    found = []
    for s in sorted(merged, key=lambda x: (x.end-x.start, x.start, x.kind)):
        pieces = [(s.start, s.end)]
        for x in found:
            pieces = [(p, q) for a, b in pieces for p, q in
                      (((a, min(b, x.start)), (max(a, x.end), b))
                       if a < x.end and x.start < b else ((a, b),)) if p < q]
        for a, b in pieces:
            value = text[a:b]
            begin = a + len(value) - len(value.lstrip())
            end = a + len(value.rstrip())
            if begin < end:
                found.append(Span(begin, end, s.kind))
    triples = [s.triple() for s in sorted(found)]
    return {"id": path.stem, "source": source, "text": text, "entities": triples,
            "adjusted": set(parsed) != set(found)}

def from_corpus(archive):
    output = {}
    with ZipFile(archive) as z:
        if z.comment.decode().strip() != CORPUS_REVISION:
            raise ValueError("Ожидалась зафиксированная версия FactRuEval: " + CORPUS_REVISION)
        paths = set(z.namelist())
        for name in ("devset", "testset"):
            output[name] = []
            for path in sorted(paths):
                p = Path(path)
                if p.parent.name != name or p.suffix != ".txt" or not p.stem.startswith("book_"):
                    continue
                output[name].append(annotation(p, z.read(path),
                    z.read(str(p.with_suffix(".spans")).replace("\\", "/")), "factrueval_"+name))
    return output

def random_digits(rng, count):
    return "".join(str(rng.randrange(10)) for _ in range(count))

def value(kind, rng, i):
    region = rng.choice(REGIONS)
    if kind == "INN":
        first = region + random_digits(rng, 2)
        body = first + random_digits(rng, 5 + (i % 2))
        s = body + v.inn_checks(body)
        return s if i % 3 else s[:4] + rng.choice((" ", "-", "\u00a0")) + s[4:]
    if kind == "PASSPORT":
        series = region + str(rng.randrange(1, 27)).zfill(2)
        number = str(rng.randrange(1, 10**6)).zfill(6)
        return (series[:2] + " " + series[2:] + " " + number, series + " " + number,
                series + number, series[:2] + "\u00a0" + series[2:] + " № " + number,
                series+"-"+number)[i % 5]
    if kind == "BANK_CARD":
        n = (16, 16, 15, 16, 19, 16, 13)[i % 7]
        prefix = "378282" if n == 15 else ("555555" if i % 2 else "411111")
        body = prefix + random_digits(rng, n - len(prefix) - 1)
        s = body + v.card_check(body)
        if i % 4 == 1:
            return s
        if n == 15:
            return " ".join((s[:4], s[4:10], s[10:]))
        return rng.choice((" ", "-", "\u00a0")).join(s[k:k+4] for k in range(0, n, 4))
    if kind == "OGRNIP":
        body = "3" + str(rng.randrange(4, 27)).zfill(2) + region + random_digits(rng, 9)
        s = body + v.ogrnip_check(body)
        return s if i % 5 else " ".join((s[:1], s[1:3], s[3:5], s[5:-1], s[-1:]))
    if kind == "IP":
        if i % 3:
            return rng.choice(("192.0.2", "198.51.100", "203.0.113")) + "." + str(rng.randrange(1, 255))
        s = ipaddress.IPv6Address(int(ipaddress.IPv6Address("2001:db8::")) +
                                  (rng.randrange(0, 65536) << 64) + rng.randrange(1, 65536))
        return s.exploded.upper() if i % 6 == 0 else s.compressed
    if kind == "EMAIL":
        name = rng.choice(("account", "client", "person", "record", "box", "contact"))
        suffix = format(rng.getrandbits(35), "x")
        host = rng.choice(("example.org", "example.com", "example.net", "sample.example.invalid",
                           "почта.example.invalid", "Example.ORG"))
        punct = ("+", ".", "_", "")[i % 4]
        return name + punct + suffix + "@" + host
    if kind == "BIRTH_CERT":
        series = rng.choice(("I", "II", "III", "IV", "V", "VI", "ХI", "IX")) + rng.choice(("-", "‑")) + rng.choice(("МЮ", "АБ", "КИ", "РК", "НЦ"))
        return series + rng.choice((" ", " № ", "\u00a0")) + str(rng.randrange(1, 10**6)).zfill(6)
    if kind == "KLADR":
        n = (13, 17, 19)[i % 3]
        main = region + random_digits(rng, n-4) + "00"
        if i % 4 == 0:
            parts = [main[:2], main[2:5], main[5:8], main[8:11]]
            if n > 13:
                parts.append(main[11:-2])
            return " ".join(parts + [main[-2:]])
        return main
    raise KeyError(kind)

def inflected(morph, string, marker, gender, case):
    picks = [p for p in morph.parse(string) if marker in p.tag and p.tag.gender == gender]
    for p in picks:
        f = p.inflect({case, "sing"})
        if f is not None:
            return f.word.capitalize()
    return None

def synthetic(split, count, seed, used_pairs, used_structured):
    rng = random.Random(seed)
    fake = Faker("ru_RU")
    fake.seed_instance(seed)
    morph = pymorphy3.MorphAnalyzer()
    rows = []
    for i in range(count):
        ti = i % len(INTRO[split])
        case = CASES[split][ti]
        gender = rng.choice(("masc", "femn"))
        female = gender == "femn"
        getter_name = fake.first_name_female if female else fake.first_name_male
        getter_last = fake.last_name_female if female else fake.last_name_male
        while True:
            nominal = getter_name()
            last_nominal = getter_last()
            last2_nominal = getter_last() if i % 9 == 0 else ""
            sig = (nominal.casefold(), last_nominal.casefold(), last2_nominal.casefold())
            if sig in used_pairs:
                continue
            first = inflected(morph, nominal, "Name", gender, case)
            last = inflected(morph, last_nominal, "Surn", gender, case)
            last2 = inflected(morph, last2_nominal, "Surn", gender, case) if last2_nominal else ""
            if first and last and (not last2_nominal or last2):
                used_pairs.add(sig)
                break
        if last2:
            last += rng.choice(["-", "‑"]) + last2
        person = (last + " " + first) if i % 3 == 0 else (first + " " + last)
        if i % 11 == 0:
            person = person.upper()
            first, last = first.upper(), last.upper()
        before, after = INTRO[split][ti].split("{person}")
        text = before + person + after.format(a="а" if female else "")
        start = len(before)
        entities = [Span(start+person.index(s), start+person.index(s)+len(s), k)
                    for s, k in ((first, "NAME"), (last, "SURNAME"))]
        # Четыре последовательных типа из восьми. Цикл даёт равное число
        # примеров каждого типа. Порядок полей в документе случайный.
        kinds = [STRUCTURED[(i+j) % 8] for j in range(4)]
        rng.shuffle(kinds)
        for j, kind in enumerate(kinds):
            variant = i*3 + j
            while True:
                s = value(kind, rng, variant)
                assert v.VALIDATORS[kind](s), (kind, s)
                key = (kind, v.digits(s) if kind not in ("EMAIL", "IP", "BIRTH_CERT") else s.lower())
                if key not in used_structured:
                    used_structured.add(key)
                    break
            field = FIELDS[kind][(i+j) % 3]
            if split == "train":
                prefix, suffix = f" {field.capitalize()}: ", "."
            elif split == "validation":
                prefix, suffix = f"\n{field} = ", ""
            else:
                prefix, suffix = (
                    (f"\n• {field} — «", "»."),
                    (f" {field.capitalize()} № ", ";"),
                    (f"\n{field.capitalize()} (значение): ", "."),
                    (f"\nЗаписано: {field} = ", " |"),
                )[(i+j) % 4]
            # Слово «значение» отделяется одним пробелом для универсальной
            # контекстной метки; внешние кавычки в разметку не включаются.
            prefix = prefix.replace(" (значение)", " значение")
            text += prefix
            entities.append(Span(len(text), len(text)+len(s), kind))
            text += s + suffix
        if split == "test":
            text += rng.choice((" Запись передана на ручную проверку.",
                "\nУведомление о результате отправлено после завершения проверки.",
                " Данные в этой версии приведены для учебного примера.",
                " Ответственный подтвердил получение документов."))
        rows.append({"id": f"synth_{split}_{i:03d}", "source": "synthetic",
                     "text": text, "entities": [x.triple() for x in sorted(entities)]})
    return rows

def negatives():
    # Структурно похожие строки, заведомо не проходящие проверку.
    output = []
    for i in range(24):
        id_body = f"5001{i+123456:06d}"
        bad_inn = id_body + str((int(v.inn_checks(id_body)) + 1) % 100).zfill(2)
        card_body = "411111" + str(123400000+i).zfill(9)
        bad_card = card_body + str((int(v.card_check(card_body)) + 1) % 10)
        og_body = "32677" + str(8210+i).zfill(9)
        bad_ogrn = og_body + str((int(v.ogrnip_check(og_body)) + 1) % 10)
        # Для учебного примера исключаем совпадение с другим
        # валидатором; такие коллизии покрыты отдельными проверками.
        if v.valid_card(bad_ogrn):
            bad_ogrn = og_body + str((int(v.ogrnip_check(og_body)) + 2) % 10)
        s = (f"В очереди проверки пакет № {i+7}. Ошибочный ИНН: {bad_inn}; "
             f"повреждённый номер карты: {bad_card}; несогласованный ОГРНИП: {bad_ogrn}. "
             f"Запись 300.0.0.{i+1} не является адресом. Файл report@invalid содержит шаблон.")
        output.append({"id": f"negative_{i:03d}", "source": "synthetic_negative",
                       "text": s, "entities": []})
    return output

def build(archive, dest):
    data = from_corpus(archive)
    dev = data["devset"]
    random.Random(42).shuffle(dev)
    splits = {"train": dev[24:], "validation": dev[:24], "test": data["testset"]}
    used_pairs, used_structured = set(), set()
    for split, count, seed in [("train", 288, 4104), ("validation", 48, 4204), ("test", 160, 4304)]:
        splits[split].extend(synthetic(split, count, seed, used_pairs, used_structured))
    splits["test"].extend(negatives())
    headings = {}
    seen_ids, seen_hashes = set(), set()
    for split, rows in splits.items():
        counts = Counter()
        sources = Counter()
        for row in rows:
            key = sha256(row["text"].encode("utf-8")).hexdigest()
            assert row["id"] not in seen_ids and key not in seen_hashes, row["id"]
            seen_ids.add(row["id"])
            seen_hashes.add(key)
            spans_from(row["entities"], row["text"])
            counts.update(s[0] for s in row["entities"])
            sources[row["source"]] += 1
            row["sha256"] = key
        headings[split] = {"texts": len(rows), "sources": dict(sources),
            "tokens": sum(len(tokens(x["text"])) for x in rows),
            "with_adjusted_corpus_spans": sum(x.get("adjusted",False) for x in rows),
            "entities": {k: counts[k] for k in KINDS}}
    assert headings["test"]["texts"] >= 100
    assert min(headings["test"]["entities"].values()) >= 50
    dest.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(dest/f"{split}.jsonl", rows)
    manifest = {"corpus_revision": CORPUS_REVISION, "corpus_split_seed": 42,
                "synthetic_seeds": {"train": 4104, "validation": 4204, "test": 4304},
                "splits": headings}
    write_json(dest/"manifest.json", manifest)
    print(manifest, flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=ROOT/"data"/"factRuEval-2016.zip")
    p.add_argument("--output", type=Path, default=ROOT/"data")
    a = p.parse_args()
    build(a.corpus, a.output)
