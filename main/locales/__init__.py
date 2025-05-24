import importlib.util
from pathlib import Path

_lang = "en"           
_cache = {}

def load(lang: str):
    global _lang, _cache
    try:
        mod = importlib.import_module(f"locales.{lang}")
        _cache = mod.STRINGS
        _lang = lang
    except ModuleNotFoundError:
        print(f"[i18n] Missing locale {lang}, falling back to English")
        mod = importlib.import_module("locales.en")
        _cache = mod.STRINGS
        _lang = "en"

def tr(key: str, **kwargs) -> str:
    txt = _cache.get(key) or _cache.get(key, key)
    return txt.format(**kwargs)

load(_lang)
