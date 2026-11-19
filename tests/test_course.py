from collections import Counter
from hashlib import sha256
import ipaddress
import json
import os
import re
import subprocess
import sys

import pytest

from pii_course.anonymizer import AnonymizationError, anonymize, canonical
from pii_course.core import KINDS, ROOT, Span, bio, read_jsonl, spans_from, tokens
from pii_course.morphology import analyzer, gender
from pii_course.recognizer import Recognizer, correct_dative_names
from pii_course.rules import resolve, rule_candidates
from pii_course import validators as v
from evaluate import compare


def in_order(text, expected):
    pos=0
    out=[]
    for k,value in expected:
        pos=text.index(value,pos)
        out.append(Span(pos,pos+len(value),k))
        pos+=len(value)
    return out


def values(r):
    return [r.text[a:b] for _,a,b in r.entities]


def single_token_matches(word, marker, g=None, c=None, n=None):
    return [p for p in analyzer().parse(word) if marker in p.tag
            and (g is None or gender(p)==g) and (c is None or p.tag.case==c)
            and (n is None or p.tag.number==n)]


def test_all_structured_invariants():
    items=[
        ("INN","500100732259"),("PASSPORT","45 12 № 621311"),
        ("BANK_CARD","4111 1111 1111 1111"),("OGRNIP","326770000000711"),
        ("IP","192.0.2.24"),("EMAIL","person@example.org"),
        ("BIRTH_CERT","III-МЮ № 123456"),("KLADR","77 000 000 123 00"),
    ]
    # Для ОГРНИП из 14-значной основы контрольная цифра проверяется отдельно.
    assert int("32677000000071") % 13 % 10 == int(items[3][1][-1])
    text=" / ".join(x for _,x in items)
    spans=in_order(text,items)
    r=anonymize(text,spans,seed=21,on_error="raise")
    out=values(r)
    assert r.text == " / ".join(out)
    assert [k for k,_,_ in r.entities] == [k for k,_ in items]
    assert all(a!=b and v.VALIDATORS[k](b) for (k,a),b in zip(items,out))
    ds=list(map(v.digits,out))
    assert ds[0][:4]=="5001" and ds[0][-2:]=="59"
    assert out[1].startswith("45 12 № ")
    assert ds[2][:6]=="411111" and ds[2][-4:]=="1111"
    assert ds[3][:5]=="32677" and ds[3][-1:]==items[3][1][-1:]
    assert out[4].startswith("192.0.2.")
    assert out[5].endswith("@example.org")
    assert out[6].startswith("III-МЮ № ")
    assert out[7].startswith("77 ") and out[7].endswith(" 00")
    assert r.stats["synonymized"]==8 and not r.warnings


def test_same_normalized_values_with_different_spelling():
    groups=[
      ("INN", "500100732259", "5001-00732259"),
      ("BANK_CARD", "4111111111111111", "4111-1111-1111-1111"),
      ("EMAIL", "person@Example.ORG", "person@example.org"),
      ("IP", "2001:0DB8:0000:0000:0000:0000:0000:0042", "2001:db8::42"),
      ("BIRTH_CERT", "II-МА № 123456", "II‑ма 123456"),
    ]
    items=[(k,x) for k,a,b in groups for x in (a,b)]
    text="; ".join(x for _,x in items)
    r=anonymize(text,in_order(text,items),seed=12,on_error="raise")
    out=values(r)
    for j,(k,a,b) in enumerate(groups):
        x,y=out[2*j:2*j+2]
        assert canonical(k,x).key == canonical(k,y).key
        assert canonical(k,x).key != canonical(k,a).key
    assert out[4].endswith("@Example.ORG") and out[5].endswith("@example.org")
    assert len(out[6].split(":"))==8 and "::" in out[7]
    assert "‑ма " in out[9]
    assert r.stats["normalized_groups"]==5


@pytest.mark.parametrize("verb,expected",[("подписала","femn"),("подписал","masc")])
def test_unisex_name_gets_context_gender_and_same_lemma(verb,expected):
    text=f"Женя {verb} документ. К Жене подошли."
    s=in_order(text,[("NAME","Женя"),("NAME","Жене")])
    r=anonymize(text,s,seed=20,on_error="raise")
    a,b=values(r)
    p=single_token_matches(a,"Name",expected,"nomn","sing")
    q=single_token_matches(b,"Name",expected,"datv","sing")
    assert {x.normal_form for x in p} & {x.normal_form for x in q}
    assert verb in r.text and r.stats["normalized_groups"]==1


def test_double_surname_and_preposition_context():
    text="Анна Иванова-Петрова подписала. К Анне Ивановой‑Петровой принесли справку."
    items=[("NAME","Анна"),("SURNAME","Иванова-Петрова"),("NAME","Анне"),
           ("SURNAME","Ивановой‑Петровой")]
    r=anonymize(text,in_order(text,items),seed=21,on_error="raise")
    x,ab,y,cd=values(r)
    assert "-" in ab and "‑" in cd
    assert r.stats["normalized_groups"]==2
    assert {p.normal_form for p in single_token_matches(x,"Name","femn","nomn","sing")} & {
           p.normal_form for p in single_token_matches(y,"Name","femn","datv","sing")}
    for a,b in zip(ab.split("-"),cd.split("‑")):
        u=single_token_matches(a,"Surn","femn","nomn","sing")
        w=single_token_matches(b,"Surn","femn","datv","sing")
        assert {p.normal_form for p in u} & {p.normal_form for p in w}


def test_plural_and_family_name():
    text="Ивановы подписали документ. К Ивановым привезли копию."
    r=anonymize(text,in_order(text,[("SURNAME","Ивановы"),("SURNAME","Ивановым")]),
                seed=34,on_error="raise")
    a,b=values(r)
    assert {p.normal_form for p in single_token_matches(a,"Surn",None,"nomn","plur")} & {
        p.normal_form for p in single_token_matches(b,"Surn",None,"datv","plur")}


def test_different_lexemes_do_not_collapse():
    text="Иван и Пётр. Повтор: Иван. Документы Петра."
    ss=in_order(text,[("NAME","Иван"),("NAME","Пётр"),("NAME","Иван"),("NAME","Петра")])
    one=anonymize(text,ss,seed=1,on_error="raise")
    two=anonymize(text,ss,seed=2,on_error="raise")
    a,b,c,d=values(one)
    assert a == c and a != b and one.text != two.text
    assert {p.normal_form for p in analyzer().parse(b) if "Name" in p.tag} & {
        p.normal_form for p in analyzer().parse(d) if "Name" in p.tag}
    assert one.text==anonymize(text,ss,seed=1,on_error="raise").text


def test_private_mail_host_is_changed_but_not_public_host():
    text="contact@family-name.example.invalid / contact@example.org"
    ss=in_order(text,[("EMAIL","contact@family-name.example.invalid"),("EMAIL","contact@example.org")])
    r=anonymize(text,ss,seed=44,on_error="raise")
    a,b=values(r)
    assert a.endswith(".example.invalid") and "family-name" not in a
    assert b.endswith("@example.org")


@pytest.mark.parametrize("value",["77 000 001 002 0010 0105",
                                  "77 000 001 002 0010 0105 0012"])
def test_kladr_building_and_flat_preserve_street_code(value):
    text="Код КЛАДР: "+value
    ss=resolve(text,rule_candidates(text))
    assert ss==in_order(text,[("KLADR",value)])
    r=anonymize(text,ss,seed=27,on_error="raise")
    output=values(r)[0]
    assert len(v.digits(output))==len(v.digits(value))
    assert output.split()[:5]==value.split()[:5]
    assert output!=value


def test_ip_v6_brackets_and_ipv4_mapped():
    text="IP: [::ffff:192.0.2.24]. Дубль: 0000:0000:0000:0000:0000:ffff:c000:0218."
    ss=resolve(text,rule_candidates(text))
    assert [text[s.start:s.end] for s in ss] == ["::ffff:192.0.2.24","0000:0000:0000:0000:0000:ffff:c000:0218"]
    r=anonymize(text,ss,seed=19,on_error="raise")
    a,b=map(ipaddress.ip_address,values(r))
    assert a==b and a.ipv4_mapped is not None and str(a.ipv4_mapped).startswith("192.0.2.")
    assert r.text.startswith("IP: [::ffff:")


def test_invalid_requisite_and_explicit_numeric_context():
    invalid="500100732258"
    text=f"ИНН: {invalid}. IP: 333.1.1.1. Паспорт: 0000000000."
    ss=resolve(text,rule_candidates(text))
    assert ss==[]
    manually_labeled=in_order(text,[("INN",invalid)])
    with pytest.raises(AnonymizationError):
        anonymize(text,manually_labeled,seed=3,on_error="raise")
    r=anonymize(text,manually_labeled,seed=3)
    assert r.stats["masked_on_error"]==1 and values(r)==["*"*12]
    good="500100732259"
    text=f"ИНН: {good}; Паспорт: 7719771234; Паспорта 50 01 007322."
    assert [s.kind for s in resolve(text,rule_candidates(text))]==["INN","PASSPORT","PASSPORT"]


def test_overlap_rules_win_and_offsets_are_half_open():
    from pii_course.rules import Candidate
    t="Адрес: Ivanov@example.org."
    email=in_order(t,[("EMAIL","Ivanov@example.org")])[0]
    weak=Span(email.start,email.start+6,"SURNAME")
    ss=resolve(t,[*rule_candidates(t),Candidate(weak,30)])
    assert ss==[email]
    lexical=tokens(t)
    tags=bio(lexical,ss)
    assert tags.count("B-EMAIL")==1 and tags.count("I-EMAIL")>1
    with pytest.raises(ValueError):
        anonymize(t,[email,weak])


def test_metrics_count_false_positives_from_o():
    text="Иван и Павел"
    gold=in_order(text,[("NAME","Иван")])
    r=compare([{"text":text,"entities":[s.triple() for s in gold]}],
             [in_order(text,[("NAME","Иван"),("NAME","Павел")])])
    name=r["typed"]["per_attribute"]["NAME"]
    assert name["tp"]==name["fp"]==1 and name["fn"]==0
    assert name["precision"]==.5 and name["recall"]==1 and name["f1"]==pytest.approx(2/3)


def test_dataset_contract_no_document_leakage():
    hashes=set()
    for split in ("train","validation","test"):
        rows=read_jsonl(ROOT/"data"/f"{split}.jsonl")
        count=Counter()
        for row in rows:
            h=sha256(row["text"].encode()).hexdigest()
            assert h==row["sha256"] and h not in hashes
            hashes.add(h)
            spans=spans_from(row["entities"],row["text"])
            count.update(x.kind for x in spans)
        assert all(count[k]>=50 for k in KINDS) if split=="test" else len(rows)>0
        if split=="test":
            assert len(rows)>=100


@pytest.fixture(scope="module")
def model():
    return Recognizer(device="cpu",threads=3)


def test_actual_model_passes_course_demo(model):
    text=(ROOT/"examples"/"demo.txt").read_text(encoding="utf-8")
    ss=model.recognize(text)
    assert set(KINDS) <= {x.kind for x in ss}
    expected=[("NAME","Женя"),("SURNAME","Иванова"),("NAME","Жене"),("SURNAME","Ивановой"),
              ("NAME","Иван"),("SURNAME","Иванов-Петров"),("NAME","Ивана"),("SURNAME","Иванова-Петрова")]
    assert set(in_order(text,expected)) <= set(ss)
    r=anonymize(text,ss,seed=19,on_error="raise")
    assert not r.warnings and r.stats["synonymized"]==len(ss)
    assert len(r.entities)==len(ss)
    assert all(0<=a<b<=len(r.text) for _,a,b in r.entities)
    assert all(a[2]<=b[1] for a,b in zip(r.entities,r.entities[1:]))

def test_dative_name_after_recipient_verb(model):
    text = "Женя Иванов передал Жене Ивановой что-то"
    expected = in_order(text, [("NAME", "Женя"), ("SURNAME", "Иванов"),
                               ("NAME", "Жене"), ("SURNAME", "Ивановой")])
    spans = model.recognize(text)
    assert spans == expected
    result = anonymize(text, spans, seed=19, on_error="raise")
    first_name, first_surname, second_name, second_surname = values(result)
    assert single_token_matches(first_name, "Name", "masc", "nomn", "sing")
    assert single_token_matches(second_name, "Name", "femn", "datv", "sing")
    masculine = single_token_matches(first_surname, "Surn", "masc", "nomn", "sing")
    feminine = single_token_matches(second_surname, "Surn", "femn", "datv", "sing")
    assert {p.normal_form for p in masculine} & {p.normal_form for p in feminine}


@pytest.mark.parametrize("text,expected", [
    ("Юра Волков любит волков", [("NAME", "Юра"), ("SURNAME", "Волков")]),
    ("Юра волков любит Волков", [("NAME", "Юра"), ("SURNAME", "волков")]),
    ("Волков любит Юру", [("SURNAME", "Волков"), ("NAME", "Юру")]),
    ("Любит Волков Юру", [("SURNAME", "Волков"), ("NAME", "Юру")]),
    ("Юра любит Волковых", [("NAME", "Юра"), ("SURNAME", "Волковых")]),
    ("Юра любит Василия Волкова", [("NAME", "Юра"), ("NAME", "Василия"), ("SURNAME", "Волкова")]),
    ("Юра волков любит Таню Орлову", [("NAME", "Юра"), ("SURNAME", "волков"),
                                        ("NAME", "Таню"), ("SURNAME", "Орлову")]),
    ("Маша Орлова любит Орлов", [("NAME", "Маша"), ("SURNAME", "Орлова")]),
])
def test_person_tags_respect_verbs_and_homonymous_nouns(model, text, expected):
    spans = model.recognize(text)
    assert spans == in_order(text, expected)
    result = anonymize(text, spans, seed=19, on_error="raise")
    assert not result.warnings
    assert "любит" in result.text.lower()
    if text.endswith("любит Волков"):
        assert result.text.endswith("любит Волков")
    if text.endswith("любит волков"):
        assert result.text.endswith("любит волков")
    assert len(result.entities) == len(expected)


def test_interactive_entrypoint_preserves_verbs_and_ordinary_nouns():
    run = subprocess.run(
        [sys.executable, str(ROOT/"main.py"), "interactive", "--device", "cpu", "--seed", "19"],
        input="2\nЮра волков любит Волков\n2\nЮра Волков любит волков\n0\n",
        text=True, encoding="utf-8", capture_output=True, check=True, timeout=60,
        cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert "Версия курсовой: 2026.11.20.2" in run.stdout
    assert "Аникита тихонов любит Волков" in run.stdout
    assert "Аникита Тихонов любит волков" in run.stdout
    assert re.findall(r"^(NAME|SURNAME)\s+\[\d+, \d+\):", run.stdout, re.M) == [
        "NAME", "SURNAME", "NAME", "SURNAME"]
    assert "BertTokenizerFast" not in run.stderr


@pytest.mark.parametrize("text,items", [
    ("Позвонил Жан Жене.", [("NAME", "Жан"), ("SURNAME", "Жене")]),
    ("Передал жене Ивановой.", [("SURNAME", "жене"), ("SURNAME", "Ивановой")]),
    ("Фамилии Жене Ивановой в списке.", [("SURNAME", "Жене"), ("SURNAME", "Ивановой")]),
])
def test_dative_name_rule_requires_recipient_context(text, items):
    original = in_order(text, items)
    assert correct_dative_names(text, original) == original


def test_released_results_trace_to_model_and_test_data():
    path=ROOT/"results"
    rec=json.loads((path/"recognition.json").read_text(encoding="utf-8"))
    sem=json.loads((path/"bertscore.json").read_text(encoding="utf-8"))
    test_digest=sha256((ROOT/"data"/"test.jsonl").read_bytes()).hexdigest()
    model_digest=sha256((ROOT/"model"/"model.safetensors").read_bytes()).hexdigest()
    assert rec["metadata"]["data_sha256"]==sem["metadata"]["data_sha256"]==test_digest
    assert rec["metadata"]["model_sha256"]==sem["metadata"]["ner_model_sha256"]==model_digest
    assert rec["bio"]["aggregate"]["macro_f1"]>=.8
    assert rec["bio"]["aggregate"]["weighted_f1"]>=.85
    assert sem["all"]["texts"]==rec["texts"]
    assert sem["all"]["synonym"]["f1"]>sem["all"]["entity_mask"]["f1"]
