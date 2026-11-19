"""Check neural person tags against ordinary words and grammatical roles."""
import re

WORDS = re.compile(r"[А-ЯЁа-яё]+(?:[-‐‑][А-ЯЁа-яё]+)*")
SPACE = re.compile(r"[ \t]+")
MARKER = {"NAME": "Name", "SURNAME": "Surn"}
CASES = {"Nom": "nomn", "Gen": "gent", "Dat": "datv", "Acc": "accs", "Ins": "ablt", "Loc": "loct"}


def _known_person(parses, kind):
    return [p for p in parses if p.is_known and MARKER[kind] in p.tag]


def _neighbor(text, atoms, index, kind):
    for other in (index - 1, index + 1):
        if not 0 <= other < len(atoms) or atoms[other][0] != kind:
            continue
        first, second = sorted((atoms[index], atoms[other]), key=lambda item: item[1])
        if SPACE.fullmatch(text[first[2]:second[1]]):
            return True
    return False


def _expected_case(node, nodes, analyzer):
    if node.rel.startswith("nsubj"):
        return "nomn"
    if node.rel == "iobj":
        return "datv"
    if node.rel == "obj":
        governor = nodes.get(node.head_id)
        if governor and any(p.tag.POS in ("VERB", "INFN") and p.tag.transitivity == "tran"
                            for p in analyzer.parse(governor.text)):
            return "accs"
    return CASES.get(node.feats.get("Case"))


def refine_person_spans(text, entities, analyzer, document):
    """Preserve offsets and spelling while correcting or excluding person tags."""
    atoms, readings = [], {}
    for group, (kind, start, end) in enumerate(entities):
        matches = list(WORDS.finditer(text, start, end))
        if kind not in MARKER or not matches:
            atoms.append([kind, start, end, group])
            continue
        for word in matches:
            options = analyzer.parse(word[0])
            names, surnames = _known_person(options, "NAME"), _known_person(options, "SURNAME")
            if not names and not surnames and any(
                    p.is_known and p.tag.POS in ("VERB", "INFN") and p.score >= .5 for p in options):
                continue
            if kind == "NAME" and surnames and not names:
                actual = "SURNAME"
            else:
                actual = kind
            atoms.append([actual, word.start(), word.end(), group])
            readings[word.start()] = (options, names, surnames, len(matches))

    for i, atom in enumerate(atoms):
        info = readings.get(atom[1])
        if info is None or atom[0] != "SURNAME":
            continue
        _, names, surnames, count = info
        if (count == 1 and not surnames and any("Fixd" not in p.tag for p in names)
                and _neighbor(text, atoms, i, "SURNAME")):
            atom[0] = "NAME"

    if not atoms:
        return []
    doc = document(text)
    nodes = {token.id: token for token in doc.tokens}
    positions = {(token.start, token.stop): token for token in doc.tokens}
    result, previous_group = [], None
    for i, (kind, start, end, group) in enumerate(atoms):
        node = positions.get((start, end))
        info = readings.get(start)
        if kind in MARKER and info is not None and node is not None:
            options = info[0]
            ordinary = [p for p in options if p.is_known and p.tag.POS == "NOUN"
                        and not any(marker in p.tag for marker in ("Name", "Surn", "Patr"))]
            expected = _expected_case(node, nodes, analyzer)
            person_fits = any(MARKER[kind] in p.tag and p.tag.case == expected for p in options)
            ordinary_fits = any(p.tag.case == expected for p in ordinary)
            full_name = _neighbor(text, atoms, i, "NAME" if kind == "SURNAME" else "SURNAME")
            if expected and ordinary_fits and not person_fits and not full_name:
                continue
        if (result and previous_group == group and result[-1][0] == kind
                and SPACE.fullmatch(text[result[-1][2]:start])):
            result[-1][2] = end
        else:
            result.append([kind, start, end])
        previous_group = group
    return result
