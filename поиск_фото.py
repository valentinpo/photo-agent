# -*- coding: utf-8 -*-
"""
Семантический поиск изображений по тексту поста для мессенджера.

Находит среди проиндексированных фотографий (индекс_фото.jsonl) те, что
лучше всего подходят под смысл текста поста. Использует нормализацию слов,
взвешивание по ключевым словам / заголовку / описанию.

Запуск:
  python поиск_фото.py "текст поста" [путь_к_папке_с_индексом ...] [-n TOP] [--json]

Пример:
  python поиск_фото.py "нужно фото уютной деревянной веранды с гирляндами для поста про осень"
"""
import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_INDEX = "индекс_фото.jsonl"


# --- Нормализация русских слов (упрощённая: срезаем окончания) ---

# Наиболее частые окончания русских словоформ, которые можно отбросить.
_SUFFIXES = [
    "ыми", "ими", "ого", "ому", "ую", "юю", "его", "ему", "ий", "ый", "ой",
    "ая", "яя", "ое", "ее", "ие", "ые", "ами", "ями", "ов", "ев", "ах",
    "ях", "ом", "ем", "ам", "ям", "ах", "ой", "ей", "ых", "их", "а", "я",
    "ы", "и", "е", "у", "ю", "о", "ь",
]

def normalize_token(word: str) -> str:
    """Приводит слово к нижнему регистру и срезает окончание (грубая стемминг)."""
    w = word.lower().strip()
    if len(w) <= 3:
        return w
    for suf in _SUFFIXES:
        if len(w) - len(suf) >= 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    """Разбивает текст на значимые токены (слова), отбрасывает стоп-слова."""
    words = re.findall(r"[а-яёa-z]{3,}", text.lower())
    stop = {
        "для", "это", "нужно", "надо", "фото", "пост", "поста", "посту",
        "или", "что", "какой", "какая", "какое", "какие", "наш", "нашёл",
        "подойдёт", "подходят", "искать", "найти", "очень", "более", "также",
        "своей", "своего", "такой", "такую", "такое", "время", "прямо",
        "сейчас", "можно", "есть", "только", "будет", "было", "этот", "эта",
        "это", "этот", "в", "на", "по", "из", "с", "за", "под", "над", "от",
        "до", "не", "ни", "и", "а", "но", "то", "же", "бы", "ли", "уж", "да",
    }
    return [w for w in words if w not in stop]


def query_tokens(query: str) -> dict[str, str]:
    """Возвращает словарь: нормализованный_токен -> исходное слово."""
    toks = {}
    for w in tokenize(query):
        n = normalize_token(w)
        if n:
            toks[n] = w
    return toks


def score_entry(entry: dict, qtokens: dict[str, str]) -> tuple[float, list[str]]:
    """Считает релевантность записи запросу. Возвращает (оценка, причины)."""
    # Нормализуем каждый полезный текст один раз
    kw_norm = [normalize_token(k) for k in entry.get("keywords", [])]
    title_norm = entry.get("title", "").lower()
    desc_norm = entry.get("description", "").lower()
    desc_norm = re.sub(r"[^а-яёa-z\s]", " ", desc_norm)

    score = 0.0
    reasons = []

    for n, orig in qtokens.items():
        # 1. Точное совпадение в keywords — самый высокий вес
        if n in kw_norm:
            score += 3.0
            reasons.append(f"ключевое слово «{orig}»")
        # 2. В заголовке — средний вес
        elif n == orig and orig.lower() in entry.get("title", "").lower() or orig.lower() in (entry.get("title") or "").lower():
            score += 2.0
            reasons.append(f"заголовок «{orig}»")
        # 3. В описании — меньший вес
        elif orig.lower() in desc_norm:
            score += 1.0
            reasons.append(f"в описании «{orig}»")

    # небольшой бонус за совпадение с несколькими токенами подряд в описании
    if score > 0:
        reasons = list(dict.fromkeys(reasons))  # убрать дубли, сохранить порядок
    return round(score, 2), reasons


def load_entries(index_paths: list[str]) -> list[tuple[dict, str]]:
    """Загружает все записи из переданных индексов. Возвращает (entry, source_path)."""
    entries = []
    for ip in index_paths:
        p = Path(ip)
        if p.is_dir():
            p = p / DEFAULT_INDEX
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append((json.loads(line), str(p)))
    return entries


def search(query: str, index_paths: list[str], top_n: int = 5) -> list[dict]:
    qtokens = query_tokens(query)
    if not qtokens:
        return []
    results = []
    for entry, src in load_entries(index_paths):
        score, reasons = score_entry(entry, qtokens)
        if score > 0:
            results.append({
                "score": score,
                "filename": entry.get("filename", ""),
                "path": entry.get("path", ""),
                "title": entry.get("title", ""),
                "reasons": reasons,
                "source": src,
            })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]


def main():
    ap = argparse.ArgumentParser(description="Поиск фото по тексту поста")
    ap.add_argument("query", help="текст поста / запрос")
    ap.add_argument("paths", nargs="*", help="папка(и) или файл(ы) индексов (по умолчанию текущая папка)")
    ap.add_argument("-n", "--top", type=int, default=5, help="сколько результатов показать (по умолчанию 5)")
    ap.add_argument("--json", action="store_true", help="вывести результат в JSON")
    args = ap.parse_args()

    paths = args.paths or ["."]
    results = search(args.query, paths, top_n=args.top)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if not results:
        print("Ничего не нашлось по запросу. Попробуйте изменить формулировку или добавить больше индексов.")
        return

    print(f"Найдено {len(results)} подходящих фотографий:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']}] {r['filename']}")
        print(f"   {r['title']}")
        print(f"   Почему: {', '.join(r['reasons'])}")
        print(f"   Путь: {r['path']}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()