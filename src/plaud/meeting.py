import hashlib
import json
import os
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path


def archive_root() -> Path:
    return Path(os.environ.get("PLAUD_ARCHIVE", Path.home() / "plaud" / "archive"))


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:40].strip("-")
    return slug or "meeting"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_existing(archive: Path, digest: str) -> Path | None:
    for mj in archive.glob("*/*/meeting.json"):
        if json.loads(mj.read_text(encoding="utf-8")).get("id") == digest:
            return mj.parent
    return None


def create_or_get(audio: Path, archive: Path | None = None,
                  when: datetime | None = None) -> Path:
    archive = archive or archive_root()
    digest = file_hash(audio)
    existing = _find_existing(archive, digest)
    if existing:
        return existing
    when = when or datetime.fromtimestamp(audio.stat().st_mtime)
    base = f"{when:%Y-%m-%d_%H%M}_{slugify(audio.stem)}"
    d, i = archive / f"{when:%Y}" / base, 2
    while d.exists():
        d, i = archive / f"{when:%Y}" / f"{base}-{i}", i + 1
    d.mkdir(parents=True)
    shutil.copy2(audio, d / f"audio{audio.suffix.lower()}")
    meta = {"id": digest, "source": str(audio),
            "start": when.isoformat(timespec="minutes"), "title": audio.stem}
    (d / "meeting.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    return d
