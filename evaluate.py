"""Оценка распознавания по BIO, типам токенов и точным интервалам."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from pii_course.core import KINDS, ROOT, bio, read_jsonl, tokens, spans_from, write_json, write_jsonl
from pii_course.recognizer import Recognizer

def counts_table(labels, tp, fp, fn):
    out = {}
    for k in labels:
        p = tp[k]/(tp[k]+fp[k]) if tp[k]+fp[k] else 0.0
        r = tp[k]/(tp[k]+fn[k]) if tp[k]+fn[k] else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        out[k] = {"tp": tp[k], "fp": fp[k], "fn": fn[k], "support": tp[k]+fn[k],
                  "precision": p, "recall": r, "f1": f}
    return out

def average(table):
    n = len(table)
    count = sum(v["support"] for v in table.values())
    result = {"labels": list(table), "support": count}
    for m in ("precision","recall","f1"):
        result["macro_"+m] = sum(v[m] for v in table.values())/max(n,1)
        result["weighted_"+m] = sum(v[m]*v["support"] for v in table.values())/max(count,1)
    tp,fp,fn = (sum(v[k] for v in table.values()) for k in ("tp","fp","fn"))
    result["micro_f1"] = 2*tp / max(2*tp+fp+fn,1)
    return result

def tag_type(tag):
    return tag[2:] if tag != "O" else "O"

def compare(rows, detected):
    c_bio = [Counter(),Counter(),Counter()]
    c_type = [Counter(),Counter(),Counter()]
    c_exact = [Counter(),Counter(),Counter()]
    n_tokens = 0
    for row, actual in zip(rows, detected, strict=True):
        expected = spans_from(row["entities"], row["text"])
        ls = tokens(row["text"])
        n_tokens += len(ls)
        x,y = bio(ls, expected), bio(ls, actual)
        for t,p in zip(x,y):
            if t == p:
                c_bio[0][t]+=1
            else:
                c_bio[1][p]+=1
                c_bio[2][t]+=1
            t,p = tag_type(t),tag_type(p)
            if t == p:
                c_type[0][t]+=1
            else:
                c_type[1][p]+=1
                c_type[2][t]+=1
        g,h = set(expected),set(actual)
        for s in g & h:
            c_exact[0][s.kind]+=1
        for s in h-g:
            c_exact[1][s.kind]+=1
        for s in g-h:
            c_exact[2][s.kind]+=1
    target_tags = tuple(i+"-"+k for k in KINDS for i in ("B","I"))
    t_type = counts_table((*KINDS,"O"), *c_type)
    t_bio = counts_table((*target_tags,"O"), *c_bio)
    t_exact = counts_table(KINDS,*c_exact)
    # Метки, которых нет ни в разметке, ни в прогнозе, не усредняются.
    def active(t):
        return {k:v for k,v in t.items() if v["tp"]+v["fp"]+v["fn"]>0}
    def non_o(t):
        return active({k:v for k,v in t.items() if k!="O"})
    return {"texts": len(rows), "tokens": n_tokens,
            "typed": {"per_attribute":t_type, "aggregate":average(non_o(t_type)),
                      "including_o":average(active(t_type))},
            "bio": {"per_tag":t_bio, "aggregate":average(non_o(t_bio)),
                    "including_o":average(active(t_bio)),
                    "absent_tags":[k for k in target_tags if k not in non_o(t_bio)]},
            "exact": {"per_attribute":t_exact, "aggregate":average(active(t_exact))},
            "empty_reference_texts": sum(not x["entities"] for x in rows),
            "false_positive_texts_without_pii": sum(not x["entities"] and bool(y)
                                                     for x,y in zip(rows,detected))}

def run(data, model, output, device="auto", threads=4):
    rows = read_jsonl(data)
    reg = Recognizer(model,device,threads)
    detected = reg.recognize_many([r["text"] for r in rows])
    result = compare(rows,detected)
    result["metadata"]={"data_file":data.name,
                        "data_sha256":sha256(data.read_bytes()).hexdigest(),
                        "model_sha256":sha256((model/"model.safetensors").read_bytes()).hexdigest(),
                        "device":reg.device,
                        "tokenization":"pii_course.core.tokens",
                        "target":"F1 weighted >= 0.85; F1 macro >= 0.80"}
    result["by_source"] = {}
    for source in dict.fromkeys(x["source"] for x in rows):
        ix=[i for i,r in enumerate(rows) if r["source"]==source]
        result["by_source"][source] = compare([rows[i] for i in ix],[detected[i] for i in ix])
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"recognition.json", result)
    write_jsonl(output/"predictions.jsonl",(
        {"id":r["id"],"entities":[s.triple() for s in actual], "source_sha256":r["sha256"]}
        for r,actual in zip(rows,detected)))
    a=result["typed"]["aggregate"]
    b=result["bio"]["aggregate"]
    print(f'Текстов: {len(rows)}; токенов: {result["tokens"]}')
    print(f'{"Атрибут":14} {"Поддержка":>9} {"Precision":>10} {"Recall":>10} {"F1":>10}')
    for k in KINDS:
        p=result["typed"]["per_attribute"][k]
        print(f'{k:14} {p["support"]:9} {p["precision"]:10.4f} {p["recall"]:10.4f} {p["f1"]:10.4f}')
    for k,v in (("Типы без O",a),("BIO без O",b),("Интервалы",result["exact"]["aggregate"])):
        print(f'{k:15} macro F1 = {v["macro_f1"]:.4f}; weighted F1 = {v["weighted_f1"]:.4f}')
    passed = b["macro_f1"] >= .80 and b["weighted_f1"] >= .85
    print("Пороги ТЗ для BIO без O:", "ВЫПОЛНЕНЫ" if passed else "НЕ ВЫПОЛНЕНЫ")
    print("Подробные результаты:",output/"recognition.json")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT/"data"/"test.jsonl")
    p.add_argument("--model", type=Path, default=ROOT/"model")
    p.add_argument("--output", type=Path, default=ROOT/"results")
    p.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    p.add_argument("--threads",type=int,default=4)
    a=p.parse_args()
    run(a.data,a.model,a.output,a.device,a.threads)
