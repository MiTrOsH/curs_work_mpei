"""Воспроизводимое сравнение BERTScore на той же тестовой выборке."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import mean
import random

from pii_course.anonymizer import anonymize, mask_spans
from pii_course.core import ROOT, read_jsonl, spans_from, write_json, write_jsonl
from pii_course.recognizer import device_for

ENCODER = "cointegrated/rubert-tiny2"
REVISION = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"
LAYERS = 3
METHODS = ("synonym", "entity_mask", "full_mask")

def encoder_path(local):
    if local is not None:
        p=Path(local).resolve()
        if not (p/"model.safetensors").exists():
            raise FileNotFoundError("Для BERTScore нужен исходный RuBERT-tiny2, не модель NER.")
        return p
    os.environ.setdefault("HF_HUB_DISABLE_XET","1")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(ENCODER,revision=REVISION, token=False,
        allow_patterns=["config.json","model.safetensors","tokenizer*","vocab.txt",
                        "special_tokens_map.json"]))

def windows(text, spans, limit):
    """Границы общие для трёх вариантов. Не разрываем сущности."""
    chunks=[]
    a=0
    while a<len(text):
        b=min(len(text),a+limit)
        if b<len(text):
            for separator in ("\n", " "):
                best=text.rfind(separator,a+limit*2//3,b)
                if best>=0:
                    b=best+1
                    break
        for s in spans:
            if s.start<b<s.end:
                b=s.end
                break
        chunks.append((a,b))
        a=b
    return chunks

def shifted(pos, source_spans, output_triples):
    return pos+sum((b-a)-(s.end-s.start)
                   for s,(_,a,b) in zip(source_spans,output_triples) if s.end<=pos)

def load_predictions(data, results):
    rows=read_jsonl(data)
    pred=read_jsonl(results/"predictions.jsonl")
    recognition=json.loads((results/"recognition.json").read_text(encoding="utf-8"))
    if sha256(data.read_bytes()).hexdigest()!=recognition["metadata"]["data_sha256"]:
        raise ValueError("Прогон распознавания сделан на другой выборке: сначала выполните evaluate.py.")
    if len(rows)!=len(pred) or any(
            a["id"]!=b["id"] or a["sha256"]!=b["source_sha256"] for a,b in zip(rows,pred)):
        raise ValueError("Предсказания не соответствуют набору: сначала выполните evaluate.py.")
    return rows,pred,recognition

def summary(rows):
    result={k:{m:mean(r[k][m] for r in rows) for m in ("precision","recall","f1")}
            for k in METHODS}
    result["texts"]=len(rows)
    result["delta_f1_synonym_entity_mask"]=result["synonym"]["f1"]-result["entity_mask"]["f1"]
    result["delta_f1_synonym_full_mask"]=result["synonym"]["f1"]-result["full_mask"]["f1"]
    result["paired_docs_syn_gt_entity_mask"]=sum(
        r["synonym"]["f1"]-r["entity_mask"]["f1"]>1e-7 for r in rows)
    result["paired_docs_syn_lt_entity_mask"]=sum(
        r["synonym"]["f1"]-r["entity_mask"]["f1"]< -1e-7 for r in rows)
    # Парный bootstrap по документам, не по фрагментам.
    rng=random.Random(48)
    ds=[r["synonym"]["f1"]-r["entity_mask"]["f1"] for r in rows]
    means=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(2000))
    result["paired_bootstrap_95pct_delta"]=[means[50],means[1949]]
    return result

def run(args):
    import torch
    from bert_score import BERTScorer
    device=device_for(args.device)
    if device=="cpu":
        torch.set_num_threads(args.threads)
    rows,predict,recognition=load_predictions(args.data,args.results)
    enc=encoder_path(args.encoder)
    evaluator=BERTScorer(model_type=str(enc),num_layers=LAYERS,lang="ru",idf=False,
                         rescale_with_baseline=False,device=device,use_fast_tokenizer=True)
    source, variants, weights, slices = [], {k: [] for k in METHODS}, [], []
    docs, diagnostics, by_kind = [], Counter(), Counter()
    for ix,(row,pred) in enumerate(zip(rows,predict)):
        text=row["text"]
        spans=spans_from(pred["entities"],text)
        result=anonymize(text,spans,seed=args.seed+ix)
        masked=mask_spans(text,spans)
        a0=len(source)
        for a,b in windows(text,spans,args.window_chars):
            new_a=shifted(a,spans,result.entities)
            new_b=shifted(b,spans,result.entities)
            source.append(text[a:b])
            variants["synonym"].append(result.text[new_a:new_b])
            variants["entity_mask"].append(masked[a:b])
            variants["full_mask"].append("*"*(b-a))
            weights.append(b-a)
        if not len(source)>a0:
            raise ValueError(f"В тесте пустой текст: {row['id']}")
        slices.append((a0,len(source)))
        diagnostics.update(result.stats)
        by_kind.update(w["kind"] for w in result.warnings)
        docs.append({"id":row["id"],"source":row["source"],"modified":result.text!=text,
                     "segments":len(source)-a0,**result.stats})
    limit=evaluator._tokenizer.model_max_length
    biggest=0
    for texts in [source,*variants.values()]:
        lengths=evaluator._tokenizer(texts,add_special_tokens=True,truncation=False,
                                     return_length=True)["length"]
        biggest=max(biggest,max(lengths, default=0))
    if biggest>limit:
        raise ValueError("Некоторые фрагменты не помещаются в энкодер. Снизьте --window-chars.")
    n=len(source)
    print(f"{len(rows)} текстов, {n} общих фрагментов, {len(METHODS)*n} пар. "
          f"Наибольшая длина входа: {biggest} субтокенов. Устройство: {device}.",flush=True)
    cands=[t for meth in METHODS for t in variants[meth]]
    pp,rr,ff=evaluator.score(cands,source*len(METHODS),batch_size=args.batch_size,verbose=args.verbose)
    metrics=dict(zip(("precision","recall","f1"),(pp.tolist(),rr.tolist(),ff.tolist())))
    for doc,(a,b) in zip(docs,slices):
        w=sum(weights[a:b])
        for i,method in enumerate(METHODS):
            offset=i*n
            doc[method]={k:sum(series[offset+j]*weights[j] for j in range(a,b))/w
                         for k,series in metrics.items()}
    report={"metadata":{"data_sha256":recognition["metadata"]["data_sha256"],
               "ner_model_sha256":recognition["metadata"]["model_sha256"],"encoder":ENCODER,
               "encoder_revision":REVISION,
               "encoder_weight_sha256":sha256((enc/"model.safetensors").read_bytes()).hexdigest(),
               "layers":LAYERS,"idf":False,"rescale_with_baseline":False,
               "bert_score_package":version("bert-score"),"transformers":version("transformers"),
               "torch":version("torch"),"device":device,"seed_base":args.seed,
               "window_chars":args.window_chars,"max_input_subtokens":biggest,
               "segments":n,"batch_size":args.batch_size},
            "all":summary(docs),"modified":summary([r for r in docs if r["modified"]]),
            "anonymization":dict(diagnostics),"masked_on_error_by_type":dict(by_kind),
            "by_source":{k:summary([r for r in docs if r["source"]==k])
                         for k in dict.fromkeys(r["source"] for r in docs)}}
    args.results.mkdir(parents=True,exist_ok=True)
    write_json(args.results/"bertscore.json",report)
    write_jsonl(args.results/"bertscore_by_text.jsonl",docs)
    for k in METHODS:
        v=report["all"][k]
        print(f'{k:14} P={v["precision"]:.6f} R={v["recall"]:.6f} F1={v["f1"]:.6f}')
    print("Обезличивание:",dict(diagnostics))
    print("Результаты:",args.results/"bertscore.json")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"data"/"test.jsonl")
    p.add_argument("--results",type=Path,default=ROOT/"results")
    p.add_argument("--encoder",type=Path,help="Путь к оригинальному RuBERT-tiny2; без него загрузить по указанной ревизии")
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--batch-size",type=int,default=24)
    p.add_argument("--threads",type=int,default=5)
    p.add_argument("--seed",type=int,default=625)
    p.add_argument("--window-chars",type=int,default=450)
    p.add_argument("--verbose",action="store_true")
    a=p.parse_args()
    if min(a.batch_size,a.threads)<1 or a.window_chars<100:
        p.error("Нужны положительные размеры пакета и число потоков, окно не короче 100 символов")
    run(a)
