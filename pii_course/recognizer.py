"""RuBERT-tiny для имён и фамилий, формальные правила для реквизитов."""
from pathlib import Path
import os
import re

from .core import ROOT, Span, WORD, tokens
from .rules import Candidate, resolve, rule_candidates

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
NER_TAGS = ("O", "B-NAME", "I-NAME", "B-SURNAME", "I-SURNAME")
NER_ID = {tag: i for i, tag in enumerate(NER_TAGS)}
MODEL_NAME = "cointegrated/rubert-tiny"
MODEL_REVISION = "5441c5ea8026d4f6d7505ec004845409f1259fb1"
MODEL_FOLDER = ROOT/"model"
MAX_LENGTH = 224
STRIDE = 40
PERSON = {"NAME", "SURNAME"}
PARTICLES = {"де", "ла", "дель", "фон", "ван", "дер", "ден", "аль", "эль", "ди", "да", "дю"}
RESERVED = {"ИНН", "ПАСПОРТ", "ОГРНИП", "КЛАДР", "СНИЛС", "ПДН", "ФИО", "РФ",
            "ООО", "ОАО", "ЗАО", "США", "МВД", "ПФР", "МЭИ"}

def encode(tokenizer, lexemes):
    encoded = tokenizer([[word for word, _, _ in lex] for lex in lexemes],
        is_split_into_words=True, return_overflowing_tokens=True,
        return_offsets_mapping=True, max_length=MAX_LENGTH, stride=STRIDE, truncation=True)
    windows = []
    for i, doc in enumerate(encoded["overflow_to_sample_mapping"]):
        ids = encoded["input_ids"][i]
        offsets = encoded["offset_mapping"][i]
        word_ids = encoded.word_ids(i)
        first = [(j,w) for j,w in enumerate(word_ids)
                 if w is not None and offsets[j][0] == 0 and (j == 0 or word_ids[j-1] != w)]
        windows.append((doc, ids, first))
    return windows

def decode(lex, scores):
    spans = []
    current = None
    for i, ((word, a, b), probs) in enumerate(zip(lex, scores)):
        tag = NER_TAGS[max(range(len(probs)), key=probs.__getitem__)]
        word_like = WORD.fullmatch(word) is not None
        if len(word) == 1:
            # Одиночный предлог в начале предложения не имя.
            # Инициал возможен только перед примыкающей точкой.
            word_like = (word_like and word.isupper() and i+1 < len(lex)
                         and lex[i+1][0] == "." and lex[i+1][1] == b)
        if word.upper() in RESERVED or not word_like or tag == "O":
            if current is not None:
                spans.append(current)
                current = None
            continue
        prefix, kind = tag.split("-",1)
        if prefix == "I" and current and current.kind == kind:
            current = Span(current.start, b, kind)
        else:
            if current:
                spans.append(current)
            current = Span(a, b, kind)
    if current:
        spans.append(current)
    return spans

def restore_surname_particles(text, spans):
    """Дополнить фамилию известной частицей при непрерывном соседстве."""
    new = []
    for s in spans:
        if s.kind == "SURNAME":
            prefix = text[max(0, s.start-55):s.start]
            m = re.search(r"\b(?:ван(?:[ \t]+(?:дер|ден|де))?|фон|де(?:[ \t]+ла)?|дель|ди|дю|аль|эль)[ \t]+$", prefix, re.I)
            if m and not any(x.end > s.start-len(m[0]) and x.start < s.start for x in spans):
                s = Span(s.start-len(m[0]),s.end,s.kind)
        new.append(s)
    return new

def correct_dative_names(text, spans):
    from .morphology import is_dative_name_pair

    corrected = list(spans)
    for i, (first, second) in enumerate(zip(spans, spans[1:])):
        if first.kind == second.kind == "SURNAME" and is_dative_name_pair(text, first, second):
            corrected[i] = Span(first.start, first.end, "NAME")
    return corrected


def refine_names(text, spans):
    from .context import document
    from .morphology import analyzer
    from .person_context import refine_person_spans

    triples = refine_person_spans(text, [span.triple() for span in spans], analyzer(), document)
    return [Span(start, end, kind) for kind, start, end in triples]

def device_for(name):
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна. Используйте --device cpu.")
    return name

def scores_many(model, tokenizer, lexemes, device, batch_size=20):
    import torch
    import numpy as np
    from transformers.data.data_collator import pad_without_fast_tokenizer_warning
    probs = [np.zeros((len(row),len(NER_TAGS)),dtype=np.float32) for row in lexemes]
    weights = [np.zeros(len(row),dtype=np.float32) for row in lexemes]
    windows = encode(tokenizer,lexemes)
    order = sorted(windows,key=lambda w:len(w[1]))
    with torch.inference_mode():
        for off in range(0,len(order),batch_size):
            batch = order[off:off+batch_size]
            padded = pad_without_fast_tokenizer_warning(
                tokenizer,[{"input_ids": ids} for _,ids,_ in batch],return_tensors="pt")
            logits = model(**{k:v.to(device) for k,v in padded.items()}).logits.float().cpu().numpy()
            for i,(row,ids,alignment) in enumerate(batch):
                for pos,word in alignment:
                    # У центра окна больше контекста, чем у края.
                    w = min(pos,len(ids)-1-pos,STRIDE//2) + 1
                    weights[row][word] += w
                    probs[row][word] += logits[i,pos]*w
    for n,w in enumerate(weights):
        probs[n] /= np.maximum(w,1)[:,None]
    return probs

class Recognizer:
    def __init__(self, path=MODEL_FOLDER, device="auto", threads=4):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        import torch
        path = Path(path)
        if not (path/"model.safetensors").exists():
            raise FileNotFoundError(f"Нет весов обученной модели: {path/'model.safetensors'}")
        self.device = device_for(device)
        if threads < 1:
            raise ValueError("Число потоков должно быть положительным")
        if self.device == "cpu":
            torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        self.model, info = AutoModelForTokenClassification.from_pretrained(
            path, local_files_only=True, use_safetensors=True, output_loading_info=True,
            torch_dtype=torch.float32)
        if (info["missing_keys"] or info["mismatched_keys"] or
                [self.model.config.id2label.get(i) for i in range(len(NER_TAGS))] != list(NER_TAGS)):
            raise ValueError("Веса или метки классификатора не соответствуют модели курсовой.")
        self.model.to(self.device).eval()

    def recognize_many(self, texts, mode="combined"):
        if mode not in ("combined", "rules", "model"):
            raise ValueError("Неизвестный режим распознавания")
        if not texts:
            return []
        ners = [[] for _ in texts]
        if mode != "rules":
            lex = [tokens(t) for t in texts]
            model_scores = scores_many(self.model,self.tokenizer,lex,self.device)
            for i, (text, words, scores) in enumerate(zip(texts, lex, model_scores)):
                spans = decode(words, scores)
                if mode == "combined":
                    spans = correct_dative_names(text, spans)
                    spans = refine_names(text, spans)
                ners[i] = restore_surname_particles(text, spans)
        return [resolve(t,[*(rule_candidates(t) if mode != "model" else ()),
                           *(Candidate(s,30) for s in neural)]) for t,neural in zip(texts,ners)]

    def recognize(self,text,mode="combined"):
        return self.recognize_many([text],mode)[0]
