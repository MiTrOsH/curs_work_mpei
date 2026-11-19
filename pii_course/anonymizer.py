"""Согласованные случайные подстановки одной функции обработки."""
from collections import defaultdict
from dataclasses import dataclass
import ipaddress
import random
import re
import string

from . import morphology as m, validators as v
from .core import Span, spans_from

PERSON = {"NAME", "SURNAME"}
# Известные сервисы и технические домены не содержат имени владельца ящика.
PUBLIC_HOSTS = frozenset({"example.org","example.net","example.com","gmail.com",
    "mail.ru","bk.ru","list.ru","inbox.ru","yandex.ru","ya.ru","outlook.com","rambler.ru"})
ROMAN = str.maketrans({"І":"I","Х":"X","С":"C","М":"M"})
MAX_DRAWS = 2048

class AnonymizationError(ValueError):
    pass

@dataclass(frozen=True)
class Form:
    kind: str
    raw: str
    key: tuple
    detail: object = None

@dataclass
class Result:
    text: str
    entities: list
    warnings: list
    stats: dict

    def output(self):
        return {"text": self.text, "entities": self.entities, "warnings": self.warnings, "stats": self.stats}

def number_style(raw, number):
    it=iter(number)
    if len(v.digits(raw)) != len(number):
        raise AnonymizationError("Изменилось количество цифр")
    return "".join(next(it) if "0" <= char <= "9" else char for char in raw)

def canonical(kind, raw):
    if not v.VALIDATORS[kind](raw):
        raise AnonymizationError("Неверная структура или контрольные разряды")
    detail=None
    if kind=="EMAIL":
        match=v.EMAIL.fullmatch(raw)
        host=v.domain_key(match["host"])
        key=(kind,match["local"],host)
        detail=match["host"]
    elif kind=="IP":
        detail=v.parse_ip(raw)
        key=(kind,detail.version,int(detail))
    elif kind=="BIRTH_CERT":
        match=v.BIRTH.fullmatch(raw)
        series=re.sub(r"\s+","",match["series"]).replace("‐","-").replace("‑","-")
        roman,cyr=series.upper().split("-",1)
        key=(kind,roman.translate(ROMAN)+"-"+cyr,match["number"])
        detail=match.start("number")
    else:
        key=(kind,v.digits(raw))
    return Form(kind,raw,key,detail)

def choose_numeric(form,rng):
    k=form.kind
    old=form.key[1]
    if k == "INN":
        n=1 if len(old)==10 else 2
        body=old[:4]+random_digits(rng,len(old)-4-n)
        computed=v.inn_checks(body)
        return body+computed if computed==old[-n:] else None
    if k == "BANK_CARD":
        # Как в ЛР2, первые 6 цифр (префикс) и последние 4 сохраняются.
        # В последних четырёх находится исходный контрольный разряд.
        val=old[:6]+random_digits(rng,len(old)-10)+old[-4:]
        return val if v.valid_card(val) else None
    if k == "PASSPORT":
        return old[:4]+str(rng.randrange(1,1000000)).zfill(6)
    if k == "OGRNIP":
        body=old[:5]+random_digits(rng,9)
        check=v.ogrnip_check(body)
        return body+check if check==old[-1] else None
    if k == "KLADR":
        if len(old) in (19,23):
            # В кодах дома и квартиры последние разряды — номер позиции,
            # а не признак актуальности. Код улицы сохраняется.
            house = str(rng.randrange(1,10000)).zfill(4)
            flat = str(rng.randrange(1,10000)).zfill(4) if len(old)==23 else ""
            return old[:15]+house+flat
        # Сохраняются код региона и признак актуальности. Нулевые
        # группы структуры по возможности остаются нулевыми.
        sections=(old[2:5],old[5:8],old[8:11],old[11:-2])
        replaceable=[i for i,p in enumerate(sections) if p and int(p)>0]
        if not replaceable:
            replaceable=[0]
        blocks=list(sections)
        i=replaceable[-1]
        blocks[i]=str(rng.randrange(1,10**len(blocks[i]))).zfill(len(blocks[i]))
        return old[:2]+"".join(blocks)+old[-2:]
    raise KeyError(k)

def random_digits(rng,n):
    return "".join(str(rng.randrange(10)) for _ in range(n))

def choose_ip(source,rng):
    if source.version==4:
        val=(int(source)&~0xff) | rng.randrange(1,255)
        return ipaddress.IPv4Address(val)
    if source.ipv4_mapped is not None:
        # IPv4-представление внутри IPv6 остаётся отображённым IPv4.
        val=(int(source)&~0xff) | rng.randrange(1,255)
        return ipaddress.IPv6Address(val)
    val=(int(source)&~((1<<64)-1)) | (int(source)&(((1<<64)-1)^0xffff)) | rng.randrange(1,65536)
    return ipaddress.IPv6Address(val)

def ip_style(source,canonical_ip):
    if canonical_ip.version==4:
        return str(canonical_ip)
    if "." in source and canonical_ip.ipv4_mapped is not None:
        base=source[:source.rfind(":")+1]
        return base+str(canonical_ip.ipv4_mapped)
    # Строим восемь шестнадцатеричных полей явно: для IPv4-mapped адресов
    # некоторые версии ipaddress выводят последние байты через точки.
    full=[format((int(canonical_ip)>>shift)&0xffff,"04x")
          for shift in range(112,-1,-16)]
    if "::" in source:
        left,right=source.split("::")
        l=left.split(":") if left else []
        r=right.split(":") if right else []
        end=len(full)-len(r)
        if any(int(chunk,16) for chunk in full[len(l):end]):
            result=canonical_ip.compressed
        else:
            result=(":".join(format(int(value,16),"x").zfill(len(original))
                  for original,value in zip(l,full))+"::"+
                  ":".join(format(int(value,16),"x").zfill(len(original))
                  for original,value in zip(r,full[end:])))
    else:
        result=":".join(format(int(value,16),"x").zfill(len(original))
                        for original,value in zip(source.split(":"),full))
    return result.upper() if any(c in "ABCDEF" for c in source) and not any(c in "abcdef" for c in source) else result

def choose_payload(form,rng):
    k=form.kind
    if k=="BIRTH_CERT":
        return str(rng.randrange(1,10**6)).zfill(6)
    if k=="EMAIL":
        token="".join(rng.choice(string.ascii_lowercase+string.digits) for _ in range(14))
        host=form.key[2]
        new=host if host in PUBLIC_HOSTS else "mail-"+token[:8]+".example.invalid"
        return "user_"+token,new
    if k=="IP":
        return choose_ip(form.detail,rng)
    return choose_numeric(form,rng)

def render_value(form,payload):
    if form.kind=="EMAIL":
        local,host=payload
        return local+"@"+(form.detail if host==form.key[2] else host)
    if form.kind=="BIRTH_CERT":
        return form.raw[:form.detail]+payload
    if form.kind=="IP":
        return ip_style(form.raw,payload)
    return number_style(form.raw,payload)

def name_options(group):
    originals={p.lemma for f in group for p in f.parts if not p.particle}
    options=[]
    for i,a in enumerate(group[0].parts):
        if a.particle:
            options.append((a.lemma,))
        else:
            candidates=[set(m.forms(f.kind,f.parts[i].gender,f.parts[i].number,f.parts[i].case))
                        for f in group]
            opts=set.intersection(*candidates)-originals
            if not opts:
                raise AnonymizationError("В словаре нет замены для всех грамматических форм")
            options.append(tuple(sorted(opts)))
    return options

def same_policy(kind,old,new):
    """Инварианты структурированной замены. Проверяется результат вставки."""
    if kind in PERSON:
        return True
    a,b=canonical(kind,old),canonical(kind,new)
    if a.key==b.key:
        return False
    if kind=="IP":
        x,y=a.detail,b.detail
        bits=8 if x.version==4 or x.ipv4_mapped is not None else 64
        return (x.version==y.version and (int(x)>>bits)==(int(y)>>bits)
                and (x.version==4 or (x.ipv4_mapped is None)==(y.ipv4_mapped is None)))
    if kind=="EMAIL":
        return b.key[2]==a.key[2] if a.key[2] in PUBLIC_HOSTS else b.key[2].endswith(".example.invalid")
    if kind=="BIRTH_CERT":
        return a.key[1]==b.key[1] and old[:a.detail]==new[:b.detail]
    x,y=a.key[1],b.key[1]
    if len(x)!=len(y) or number_style(old,y)!=new:
        return False
    return {
        "INN": x[:4]==y[:4] and x[-(1 if len(x)==10 else 2):]==y[-(1 if len(x)==10 else 2):],
        "PASSPORT": x[:4]==y[:4],
        "BANK_CARD": x[:6]==y[:6] and x[-4:]==y[-4:],
        "OGRNIP": x[:5]==y[:5] and x[-1]==y[-1],
        "KLADR": (x[:15]==y[:15] if len(x) in (19,23) else
                  x[:2]==y[:2] and x[-2:]==y[-2:]),
    }[kind]

def anonymize(text, spans, *, seed=None, on_error="mask"):
    if on_error not in ("mask","raise"):
        raise ValueError("Допустимы режимы mask и raise")
    spans=spans_from([s.triple() if isinstance(s,Span) else s for s in spans],text)
    if seed is not None and not isinstance(seed,int):
        raise ValueError("seed должен быть целым числом")
    rng=random.SystemRandom() if seed is None else random.Random(seed)
    person=[s for s in spans if s.kind in PERSON]
    prepared, warnings = {}, {}
    for s in spans:
        try:
            prepared[s] = (m.profile(text,s,person) if s.kind in PERSON
                           else canonical(s.kind,text[s.start:s.end]))
        except (ValueError, TypeError) as exc:
            if on_error=="raise":
                raise AnonymizationError(f"{s.kind} [{s.start},{s.end}): {exc}") from exc
            warnings[s]=str(exc)
    profiles=[s for s in prepared if s.kind in PERSON]
    for s,f in zip(profiles,m.reconcile([prepared[s] for s in profiles])):
        prepared[s]=f
    groups=defaultdict(list)
    for s,form in prepared.items():
        groups[form.key].append(s)
    reserved=set(groups)
    used=set()
    replacements={}
    for key,group_spans in groups.items():
        fs=[prepared[s] for s in group_spans]
        first=fs[0]
        try:
            opts=name_options(fs) if first.kind in PERSON else ()
            for _ in range(MAX_DRAWS):
                if first.kind in PERSON:
                    payload=tuple(rng.choice(values) for values in opts)
                    new_key=(key[0],payload,key[2])
                    values=[m.render_name(f,payload) for f in fs]
                else:
                    payload=choose_payload(first,rng)
                    if payload is None:
                        continue
                    values=[render_value(f,payload) for f in fs]
                    if not all(same_policy(first.kind,f.raw,out) for f,out in zip(fs,values)):
                        continue
                    new_key=canonical(first.kind,values[0]).key
                    if any(canonical(first.kind,out).key!=new_key for out in values):
                        continue
                if new_key in reserved or new_key in used or any(
                        m.fold(f.raw)==m.fold(out) for f,out in zip(fs,values)):
                    continue
                used.add(new_key)
                replacements.update(zip(group_spans,values))
                break
            else:
                raise AnonymizationError("Не удалось сгенерировать отличающуюся допустимую замену")
        except (ValueError,TypeError,KeyError) as exc:
            if on_error=="raise":
                raise AnonymizationError(f"{first.kind}: {exc}") from exc
            for s in group_spans:
                warnings[s]=str(exc)
    result, offsets=[],[]
    at=out_at=0
    for s in spans:
        unchanged=text[at:s.start]
        out=replacements.get(s,"*"*(s.end-s.start))
        result.extend((unchanged,out))
        out_at+=len(unchanged)
        offsets.append([s.kind,out_at,out_at+len(out)])
        out_at+=len(out)
        at=s.end
    result.append(text[at:])
    return Result("".join(result),offsets,
                  [{"kind":s.kind,"start":s.start,"end":s.end,"reason":message}
                   for s,message in sorted(warnings.items())],
                  {"recognized":len(spans),"synonymized":len(replacements),"masked_on_error":len(warnings),
                   "normalized_groups":len(groups),
                   "guessed_morphology":sum(any(p.inferred for p in f.parts) for f in prepared.values()
                                            if isinstance(f,m.Profile))})

def mask_spans(text,spans):
    pieces=[]
    last=0
    for s in spans:
        pieces.extend((text[last:s.start],"*"*(s.end-s.start)))
        last=s.end
    pieces.append(text[last:])
    return "".join(pieces)
