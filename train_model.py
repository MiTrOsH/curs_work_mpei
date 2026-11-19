"""Дообучить RuBERT-tiny. Выбор эпохи по отдельной validation.jsonl."""
import argparse
from collections import Counter
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
from pii_course.core import ROOT, bio, read_jsonl, spans_from, tokens, write_json
from pii_course.recognizer import (
    MODEL_NAME, MODEL_REVISION, NER_TAGS, NER_ID, decode, device_for, encode, scores_many,
)

BASES = {
    "tiny": (MODEL_NAME, MODEL_REVISION),
    "tiny2": ("cointegrated/rubert-tiny2", "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"),
}

def read_labeled(path):
    rows = read_jsonl(path)
    words = [tokens(r["text"]) for r in rows]
    tags = [bio(lex, [s for s in spans_from(r["entities"],r["text"]) if s.kind in ("NAME","SURNAME")])
            for lex,r in zip(words,rows)]
    return words,tags

def type_tag(tag):
    return tag[2:] if tag != "O" else tag

def validation(model,tok,lexemes,targets,device,batch):
    model.eval()
    tp,fp,fn = Counter(),Counter(),Counter()
    for lex,prob,true in zip(lexemes,scores_many(model,tok,lexemes,device,batch),targets):
        predicted = bio(lex,decode(lex,prob))
        for a,b in zip(map(type_tag,true),map(type_tag,predicted)):
            if a==b:
                tp[a]+=1
            else:
                fp[b]+=1
                fn[a]+=1
    val = {k: 2*tp[k]/max(1,2*tp[k]+fn[k]+fp[k]) for k in ("NAME","SURNAME")}
    return val, sum(val.values())/len(val)

def train(args):
    device = device_for(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if device == "cpu":
        torch.set_num_threads(args.threads)
    base_name, base_revision = BASES[args.architecture]
    base = args.base or base_name
    kwargs = {"revision": base_revision} if base == base_name else {"local_files_only": True}
    tok = AutoTokenizer.from_pretrained(base,use_fast=True,**kwargs)
    model = AutoModelForTokenClassification.from_pretrained(
        base,use_safetensors=True,label2id=NER_ID,id2label=dict(enumerate(NER_TAGS)),**kwargs)
    model.to(device)
    train_lex,train_labels = read_labeled(args.data/"train.jsonl")
    val_lex,val_labels = read_labeled(args.data/"validation.jsonl")
    windows = []
    for d,ids,alignment in encode(tok,train_lex):
        labels = [-100]*len(ids)
        for pos,w in alignment:
            labels[pos] = NER_ID[train_labels[d][w]]
        if any(x != -100 for x in labels):
            windows.append((ids,labels))
    print("device",device,"train windows",len(windows),"validation texts",len(val_lex),flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    steps = (len(windows)+args.batch_size-1)//args.batch_size*args.epochs
    schedule = get_linear_schedule_with_warmup(optimizer,int(steps*.05),steps)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,3.,4.,3.,4.],device=device),
                                          ignore_index=-100)
    best, log = -1, []
    for epoch in range(1,args.epochs+1):
        stamp = time.monotonic()
        model.train()
        # Случайное перемешивание; сортировка по длине лишь внутри блока.
        rng.shuffle(windows)
        order = []
        block = args.batch_size*8
        for offset in range(0,len(windows),block):
            part = sorted(windows[offset:offset+block],key=lambda r:len(r[0]))
            batch_group = [part[i:i+args.batch_size] for i in range(0,len(part),args.batch_size)]
            rng.shuffle(batch_group)
            order.extend(batch_group)
        loss_total = 0
        for batch in order:
            inputs = tok.pad([{"input_ids":ids} for ids,_ in batch],return_tensors="pt")
            width = inputs["input_ids"].shape[1]
            labels = torch.tensor([tags+[-100]*(width-len(tags)) for _,tags in batch],device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**{k:x.to(device) for k,x in inputs.items()}).logits
            loss = criterion(logits.transpose(1,2),labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            schedule.step()
            loss_total += loss.item()
        per_type,score = validation(model,tok,val_lex,val_labels,device,args.batch_size)
        val = {"epoch":epoch,"train_loss":loss_total/len(order),"val_type_macro_f1":score,
               "validation_f1":per_type,"seconds":round(time.monotonic()-stamp,1)}
        print(val,flush=True)
        log.append(val)
        if score > best:
            best,best_epoch = score,epoch
            args.output.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(args.output, safe_serialization=True)
            tok.save_pretrained(args.output)
    write_json(args.output/"training.json", {
        "base_model":base_name,"revision":base_revision,"torch":torch.__version__,
        "train_seed":args.seed,"epochs":args.epochs,"batch_size":args.batch_size,"learning_rate":args.lr,
        "best_epoch":best_epoch,"best_validation_f1":best,"validation_metric":"macro token-type F1 for NAME and SURNAME only",
        "train_windows":len(windows),"device":device,"storage_dtype":args.weight_format,"log":log,
    })
    if args.weight_format == "float16":
        from safetensors.torch import load_file, save_file
        weights = args.output/"model.safetensors"
        state = load_file(weights)
        staged = weights.with_suffix(".tmp")
        save_file({k:v.half().contiguous() if v.is_floating_point() else v
                   for k,v in state.items()},staged,metadata={"format":"pt"})
        staged.replace(weights)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture",choices=tuple(BASES),default="tiny2")
    p.add_argument("--base", help="Локальная копия исходной модели (необязательно)")
    p.add_argument("--data", type=Path, default=ROOT/"data")
    p.add_argument("--output",type=Path, default=ROOT/"model")
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--lr",type=float,default=1.8e-4)
    p.add_argument("--seed",type=int,default=142)
    p.add_argument("--threads",type=int,default=6)
    p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    p.add_argument("--weight-format",choices=("float32","float16"),default="float16")
    a = p.parse_args()
    if min(a.epochs,a.batch_size,a.threads)<1:
        p.error("Число эпох, размер пакета и число потоков должны быть положительными")
    train(a)
