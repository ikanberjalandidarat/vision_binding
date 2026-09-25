import json
from pathlib import Path
from functools import lru_cache
from jsonschema import Draft202012Validator


@lru_cache()
def validator(name):
    schema = json.loads((Path(__file__).parent/'schemas'/f'{name}-1.0.json').read_text())
    return Draft202012Validator(schema)


def check(name, value):
    json.dumps(value, allow_nan=False)
    validator(name).validate(value)
