import re
from .scenes import COLORS, TYPES

PARSER_VERSION = "pair_v1"
# Empty until isolated stimulus recognition validates synonyms; frozen per run.
DEFAULT_ALIASES = {}


def parse(raw, aliases=None, task="pair"):
    words = re.findall(r"[a-z]+", raw.lower())
    if aliases:
        words = [aliases.get(w, w) for w in words]
    expected = 2 if task == "pair" else 1
    valid = len(words) == expected
    if task == "pair":
        valid = valid and words[0] in COLORS and words[1] in TYPES
        pair = dict(zip(("color", "type"), words)) if valid else None
    else:
        valid = valid and words[0] in (COLORS if task == "color" else TYPES)
        pair = {task: words[0]} if valid else None
    return {"parsed": pair, "status": "valid" if valid else "invalid", "parser_version": PARSER_VERSION}


def attributes(obj):
    return {"color": obj["color"], "type": obj["type"]}


def score(raw, recipient, donor, target, selected, aliases=None, task="pair"):
    parsed = parse(raw, aliases, task)
    pred = parsed["parsed"]
    def match(obj):
        return pred is not None and all(obj[k] == v for k, v in pred.items())
    flags = {"recipient_target": match(recipient[target]), "recipient_at_donor_address": match(recipient[selected]), "donor_at_target": match(donor[target]), "donor_at_donor_address": match(donor[selected])}
    recombination = pred is not None and task == "pair" and pred['color'] in {o['color'] for o in recipient} and pred['type'] in {o['type'] for o in recipient} and not any(match(o) for o in recipient)
    category = next((k for k, v in flags.items() if v), "recipient_recombination" if recombination else "other_recognized" if pred else "invalid")
    return {**parsed, "flags": flags, "outcome": category}
