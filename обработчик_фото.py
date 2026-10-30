# -*- coding: utf-8 -*-
"""
Обработчик фото для агента «подбери изображение по посту».

Что делает:
  1) сканирует папку на фото (JPG/JPEG/PNG/WebP), включая подпапки
  2) описывает каждое фото через Yandex AI Studio Vision (модель qwen3.6-35b-a3b):
     полное описание на русском + ключевые слова + заголовок для переименования
  3) переименовывает файлы в понятные названия (например «закат_на_море.jpg»)
  4) записывает описание в EXIF (JPG/JPEG) и в общий индекс JSONL + CSV
     (для агента — поиск изображения по смыслу)

Конфигурация — файл .env рядом со скриптом:
  YANDEX_API_KEY=<API-ключ сервисного аккаунта, AQVN...>
  YANDEX_FOLDER_ID=<идентификатор каталога, b1g...>
  YANDEX_MODEL=qwen3.6-35b-a3b   (мультимодальная vision-модель Яндекса)

Запуск:
  python обработчик_фото.py <папка_с_фото> [--dry-run]
"""
import argparse
import base64
import csv
import json
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
INDEX_JSONL = "индекс_фото.jsonl"
INDEX_CSV = "индекс_фото.csv"
YANDEX_BASE = "https://ai.api.cloud.yandex.net/v1"

# ---------------- Конфигурация из .env ----------------

def load_env(path: Path = Path(__file__).parent / ".env") -> dict:
    cfg = {}
    env_path = path if Path(path).exists() else Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    for k in ["YANDEX_API_KEY", "YANDEX_FOLDER_ID", "YANDEX_MODEL"]:
        cfg.setdefault(k, os.environ.get(k, ""))
    return cfg

CONFIG = load_env()

def get_model() -> str:
    return CONFIG.get("YANDEX_MODEL", "qwen3.6-35b-a3b")


# ---------------- Скан, EXIF, индекс ----------------

def scan_folder(folder: str) -> list[Path]:
    out = []
    for f in sorted(Path(folder).rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            out.append(f)
    return out


def write_exif_description(path: Path, description: str, title: str = "") -> bool:
    if path.suffix.lower() not in {".jpg", ".jpeg"}:  # PNG/WebP EXIF не пишем
        return False
    if not HAS_PIEXIF:
        return False
    try:
        exif_dict = piexif.load(str(path))
        if title:
            exif_dict["0th"][270] = title.encode("utf-8")  # ImageDescription
        comment = description or ""
        exif_dict["Exif"][37510] = b"ASCII\x00\x00\x00" + comment.encode("utf-8")
        piexif.insert(piexif.dump(exif_dict), str(path))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] EXIF {path.name}: {e}")
        return False


def _append_jsonl(entry: dict, index_path: str = INDEX_JSONL):
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _rebuild_csv(index_path: str = INDEX_JSONL, csv_path: str = INDEX_CSV):
    if not os.path.exists(index_path):
        return
    rows = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return
    fieldnames = ["filename"]
    for r in rows:
        for k in r:
            if k != "filename" and k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _slugify(text: str, maxlen: int = 40) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-zа-я0-9\s]+", " ", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:maxlen].rstrip("_")


# ---------------- GigaChat vision (SDK) ----------------

def describe_image(path: Path) -> dict:
    """Отправляет фото в Yandex AI Studio vision (qwen3.6) и получает JSON.

    Ожидаемый ответ модели — строка-объект:
      {"title": ..., "description": ..., "keywords": [...]}
    """
    api_key = CONFIG.get("YANDEX_API_KEY", "")
    folder_id = CONFIG.get("YANDEX_FOLDER_ID", "")
    if not api_key or not folder_id:
        raise RuntimeError("В .env не заданы YANDEX_API_KEY / YANDEX_FOLDER_ID")

    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else \
           ("image/png" if path.suffix.lower() == ".png" else "image/webp")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")

    system = (
        "Ты — система распознавания изображений для фото-архива. "
        "Отвечай ТОЛЬКО на русском языке. Внимательно посмотри на фото и опиши "
        "ТОЛЬКО то, что реально видно. НЕ выдумывай людей, животных, предметы "
        "или действия, которых нет. Верни ТОЛЬКО валидный JSON без пояснений в формате: "
        '{"title":"короткий заголовок из 3-6 слов на русском языке, понятный сюжет, без вымысла",'
        '"description":"полное описание (4-7 предложений на русском): главный объект, детали, '
        'фон, цвета, настроение, есть ли люди и что они делают",'
        '"keywords":["тег1","тег2","тег3"]}. '
        "keywords — 4-8 коротких тегов на русском для поиска по смыслу."
    )
    full_model = f"gpt://{folder_id}/{get_model()}/latest"
    payload = {
        "model": full_model,
        "instructions": system,
        "input": [
            {"role": "user", "content": [
                {"type": "input_text", "text": "Опиши это изображение."},
                {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
            ]},
        ],
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        YANDEX_BASE + "/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"API-Key {api_key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Yandex API {e.code}: {e.read().decode()[:300]}") from e

    # Извлекаем тексты из output (игнорируем reasoning, берём output_text)
    texts = []
    for o in data.get("output", []):
        for c in o.get("content") or []:
            t = c.get("text")
            if t:
                texts.append(t)
    content = "\n".join(texts).strip()
    if not content:
        content = json.dumps(data, ensure_ascii=False)[:500]

    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.S)
    if m:
        content = m.group(1)
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        obj = {"title": path.stem, "description": content, "keywords": []}
    return {
        "title": str(obj.get("title", path.stem)),
        "description": str(obj.get("description", content)),
        "keywords": [str(k) for k in obj.get("keywords", [])],
    }


# ---------------- Главная логика ----------------

def _load_existing(index_path: str) -> dict[str, dict]:
    """Читает существующий индекс и возвращает словарь {path: entry}."""
    out = {}
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        e = json.loads(line)
                        if "path" in e:
                            out[e["path"]] = e
                    except json.JSONDecodeError:
                        pass
    return out


def process_folder(folder: str, dry_run: bool = False, skip_existing: bool = False,
                   new_index: bool = False) -> None:
    photos = scan_folder(folder)
    if not photos:
        print("Нет фотографий в папке.")
        return
    print(f"Найдено фото: {len(photos)}\n")
    index_path = str(Path(folder) / INDEX_JSONL)
    csv_path = str(Path(folder) / INDEX_CSV)

    # Читаем прежний индекс (для skip_existing и дедупликации новых проходов)
    existing = {} if new_index else _load_existing(index_path)

    succeeded = failed = renamed = skipped = 0

    for i, orig in enumerate(photos, 1):
        print(f"[{i}/{len(photos)}] {orig.name}")
        # Если файл уже описан и не просили заново — пропускаем
        if skip_existing and str(orig) in existing:
            print(f"   = пропущен (уже в индексе)")
            skipped += 1
            continue
        try:
            info = describe_image(orig)
        except Exception as e:  # noqa: BLE001
            print(f"   [error] {e}")
            failed += 1
            if not dry_run:
                existing[str(orig)] = {
                    "filename": orig.name, "path": str(orig), "title": "",
                    "description": f"[не удалось описать: {e}]",
                    "keywords": [], "notes": "describe_failed",
                }
            continue

        target = orig
        # Переименовываем файл, если в его имени нет русских букв — это «сырое»
        # имя (UUID, img_NN и т.п.). Осмысленные кириллические имена не трогаем.
        stem_has_uuid = not re.search(r"[а-яёА-ЯЁ]", orig.stem)
        base = _slugify(info["title"]) if info["title"] else ""
        if base and stem_has_uuid:
            new_name = base + orig.suffix.lower()
            candidate = orig.with_name(new_name)
            n = 2
            while candidate.exists() and candidate != orig:
                candidate = orig.with_name(f"{base}_{n}{orig.suffix.lower()}")
                n += 1
            if candidate != orig:
                if not dry_run:
                    orig.rename(candidate)
                else:
                    print(f"   [dry-run] {orig.name} -> {candidate.name}")
                target = candidate
                renamed += 1

        if not dry_run:
            write_exif_description(target, info["description"], info["title"])
            # Кладём/обновляем запись по ключу path (дедупликация)
            existing[str(target)] = {
                "filename": target.name,
                "path": str(target),
                "title": info["title"],
                "description": info["description"],
                "keywords": info["keywords"],
                "notes": "",
            }
        succeeded += 1
        print(f"   ✓ {info['title']}")

    if not dry_run:
        # Пересобираем JSONL и CSV начисто из дедуплицированного словаря
        with open(index_path, "w", encoding="utf-8") as f:
            for e in existing.values():
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        _rebuild_csv(index_path, csv_path)
    print(f"\nГотово: обработано {succeeded}, пропущено {skipped}, "
          f"ошибок {failed}, переименовано {renamed}.")
    if not dry_run:
        print(f"Индекс: {index_path}")
        print(f"CSV:    {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="Обработка фотографий через Yandex vision")
    ap.add_argument("folder", nargs="?", default=".", help="папка с фото")
    ap.add_argument("--dry-run", action="store_true", help="не менять файлы, только показать")
    ap.add_argument("--skip-existing", action="store_true",
                    help="не переописывать фото, уже присутствующие в индексе")
    ap.add_argument("--new-index", action="store_true",
                    help="начать индекс заново (игнорировать прежний)")
    args = ap.parse_args()
    process_folder(args.folder, dry_run=args.dry_run,
                   skip_existing=args.skip_existing, new_index=args.new_index)


if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    main()