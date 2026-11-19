"""Формат и контрольные разряды. Без обращения к реестрам и сети."""
import ipaddress
import re

SPACE = " \u00a0\u202f\t"
SEP = "[" + SPACE + "\u2010\u2011-]"
NUMERIC = re.compile(r"[0-9](?:[0-9" + SPACE + r"\u2010\u2011-]*[0-9])?\Z")
PASSPORT = re.compile(
    r"(?:[0-9]{2}" + SEP + r"?[0-9]{2})" + SEP + r"*(?:№[" + SPACE +
    r"]*)?[0-9]{6}\Z")
BIRTH = re.compile(
    r"(?P<series>[IVXLCDMІХСМivxlcdmіхсм]{1,7}[" + SPACE + r"]*[\u2010\u2011-][" + SPACE +
    r"]*[А-ЯЁа-яё]{2})[" + SPACE + r"]*(?:№[" + SPACE + r"]*)?(?P<number>[0-9]{6})\Z")
LOCAL = r"[A-Za-z0-9!#$%&'*+/=?^_\x60{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_\x60{|}~-]+)*"
DOMAIN_LABEL = r"[0-9A-Za-zА-ЯЁа-яё](?:[0-9A-Za-zА-ЯЁа-яё-]{0,61}[0-9A-Za-zА-ЯЁа-яё])?"
EMAIL = re.compile(r"(?P<local>" + LOCAL + r")@(?P<host>" + DOMAIN_LABEL +
                   r"(?:\." + DOMAIN_LABEL + r")+)")
INN_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
INN_12A = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
INN_12B = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)

def digits(value):
    return re.sub(r"[^0-9]", "", value)

def inn_checks(body):
    def check(values, weights):
        return str(sum(int(x)*w for x, w in zip(values, weights)) % 11 % 10)
    if len(body) == 9:
        return check(body, INN_10)
    if len(body) == 10:
        a = check(body, INN_12A)
        return a + check(body + a, INN_12B)
    raise ValueError("В основе ИНН должно быть 9 или 10 цифр")

def valid_inn(value):
    if not NUMERIC.fullmatch(value):
        return False
    s = digits(value)
    n = {10: 1, 12: 2}.get(len(s))
    return n is not None and int(s[:2]) >= 1 and len(set(s)) > 1 and inn_checks(s[:-n]) == s[-n:]

def luhn_total(number):
    total = 0
    for pos, digit in enumerate(reversed(number)):
        d = int(digit) * (2 if pos % 2 else 1)
        total += d - 9 if d > 9 else d
    return total

def card_check(body):
    return str((-luhn_total(body + "0")) % 10)

def valid_card(value):
    if not NUMERIC.fullmatch(value):
        return False
    s = digits(value)
    return 13 <= len(s) <= 19 and s[0] != "0" and len(set(s)) > 1 and luhn_total(s) % 10 == 0

def ogrnip_check(body):
    return str(int(body) % 13 % 10)

def valid_ogrnip(value):
    if not NUMERIC.fullmatch(value):
        return False
    s = digits(value)
    return len(s) == 15 and s[0] == "3" and s[3:5] != "00" and ogrnip_check(s[:-1]) == s[-1]

def valid_passport(value):
    if not PASSPORT.fullmatch(value):
        return False
    s = digits(value)
    return len(s) == 10 and s[:2] != "00" and s[-6:] != "000000"

def valid_birth(value):
    m = BIRTH.fullmatch(value)
    return bool(m and m["number"] != "000000")

def valid_kladr(value):
    if not NUMERIC.fullmatch(value):
        return False
    s = digits(value)
    return len(s) in (13, 17, 19, 23) and s[:2] != "00"

def parse_ip(value):
    if "%" in value:
        raise ValueError("IP с идентификатором зоны вне поддерживаемого формата")
    return ipaddress.ip_address(value)

def valid_ip(value):
    try:
        parse_ip(value)
        return True
    except ValueError:
        return False

def domain_key(domain):
    host = domain.encode("idna").decode("ascii").lower()
    labels = host.split(".")
    if (len(host) > 253 or len(labels) < 2 or len(labels[-1]) < 2 or labels[-1].isdigit()
            or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", p) for p in labels)):
        raise ValueError("Некорректный домен")
    return host

def valid_email(value):
    m = EMAIL.fullmatch(value)
    if not m or len(m["local"]) > 64:
        return False
    try:
        return len(m["local"]) + 1 + len(domain_key(m["host"])) <= 254
    except ValueError:
        return False

VALIDATORS = {"INN": valid_inn, "PASSPORT": valid_passport, "BANK_CARD": valid_card,
              "OGRNIP": valid_ogrnip, "IP": valid_ip, "EMAIL": valid_email,
              "BIRTH_CERT": valid_birth, "KLADR": valid_kladr}
