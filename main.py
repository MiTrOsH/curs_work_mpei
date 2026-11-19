"""Распознавание и синонимическое обезличивание персональных данных."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

from pii_course import __version__
from pii_course.anonymizer import AnonymizationError, anonymize
from pii_course.core import bio, tokens
from pii_course.recognizer import MODEL_FOLDER, Recognizer

@lru_cache(maxsize=4)
def recognizer(model=MODEL_FOLDER, device="auto"):
    return Recognizer(model, device)

def identify(text, *, model=MODEL_FOLDER, device="auto"):
    return [s.triple() for s in recognizer(Path(model).resolve(),device).recognize(text)]

def anon(text, *, model=MODEL_FOLDER, device="auto", seed=None):
    """Вернуть пару: новый текст и тройки [тип, начало, конец] в новом тексте."""
    spans=recognizer(Path(model).resolve(),device).recognize(text)
    result=anonymize(text,spans,seed=seed,on_error="raise")
    return result.text,result.entities

def print_entities(text,triples):
    if not triples:
        print("Целевые атрибуты не найдены.")
    for kind,a,b in triples:
        print(f"{kind:13} [{a}, {b}): {text[a:b]}")

def run_one(reg,text,operation,seed=None,strict=False,show_bio=False,as_json=False):
    spans=reg.recognize(text)
    if operation=="detect":
        output={"text":text,"entities":[s.triple() for s in spans]}
        if show_bio:
            lex=tokens(text)
            output["bio"]=[{"token":word,"start":a,"end":b,"tag":tag}
                           for (word,a,b),tag in zip(lex,bio(lex,spans))]
        if not as_json:
            print_entities(text,output["entities"])
            if show_bio:
                print("BIO:", " ".join(f"{t['token']}/{t['tag']}" for t in output["bio"]))
    else:
        result=anonymize(text,spans,seed=seed,on_error="raise" if strict else "mask")
        output=result.output()
        if not as_json:
            print(result.text)
            print("\nИнтервалы в обезличенном тексте:")
            print_entities(result.text,result.entities)
            for warning in result.warnings:
                print(f"Маска вместо замены: {warning['kind']} "
                      f"[{warning['start']}, {warning['end']}): {warning['reason']}")
    if as_json:
        print(json.dumps(output,ensure_ascii=False,indent=2))

def interactive(reg,seed=None,strict=False):
    print(f"Файл программы: {Path(__file__).resolve()}")
    print(f"Устройство: {reg.device}. Распознавание: RuBERT-tiny2 и правила.")
    while True:
        try:
            print("\n1 — распознавание    2 — обезличивание    0 — выход")
            choice=input("Режим: ").strip()
            if choice in ("0","/exit"):
                break
            if choice not in ("1","2"):
                print("Введите 1, 2 или 0.")
                continue
            text=input("Текст одной строкой (или /exit): ")
            if text.strip()=="/exit":
                break
            if not text.strip():
                print("Введите непустой текст.")
                continue
            try:
                run_one(reg,text,"detect" if choice=="1" else "anon",seed,strict)
            except AnonymizationError as exc:
                print(f"Обезличивание прервано: {exc}")
        except (EOFError,KeyboardInterrupt):
            print()
            break

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version",action="version",version=f"Курсовая, чётный вариант 6: {__version__}")
    p.add_argument("mode",choices=("detect","anon","interactive"),nargs="?",default="interactive")
    inputs=p.add_mutually_exclusive_group()
    inputs.add_argument("--text")
    inputs.add_argument("--file",type=Path)
    p.add_argument("--model",type=Path,default=MODEL_FOLDER)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--seed",type=int,help="Фиксированный генератор для повторения демонстрации")
    p.add_argument("--strict",action="store_true",help="Остановиться, если допустимую замену построить нельзя")
    p.add_argument("--bio",action="store_true",help="Показать BIO-разметку в режиме detect")
    p.add_argument("--json",action="store_true",help="Вывести JSON в режимах detect и anon")
    a=p.parse_args()
    if a.mode!="interactive" and a.text is None and a.file is None:
        p.error("Укажите --text или --file; для ручного ввода используйте interactive")
    if a.mode=="interactive" and (a.text is not None or a.file is not None):
        p.error("Для --text и --file используйте detect или anon")
    try:
        reg=recognizer(a.model.resolve(),a.device)
        if a.mode=="interactive":
            interactive(reg,a.seed,a.strict)
        else:
            if a.file:
                with a.file.open(encoding="utf-8-sig",newline="") as f:
                    text=f.read()
            else:
                text=a.text
            run_one(reg,text,a.mode,a.seed,a.strict,a.bio,a.json)
    except (ValueError,OSError) as exc:
        p.exit(1,f"Ошибка: {exc}\n")

if __name__=="__main__":
    main()
