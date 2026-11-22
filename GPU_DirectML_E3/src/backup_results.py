from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

from .config import OUTPUT_DIR, ROOT


def main() -> int:
    candidates = [
        path for path in OUTPUT_DIR.glob("E3_DirectML_*")
        if path.is_dir() and (path / "resumo_execucao.json").exists()
    ]
    if not candidates:
        print("Ainda não encontrei uma execução finalizada em outputs.")
        return 1
    source = max(candidates, key=lambda path: path.stat().st_mtime)
    backup_dir = ROOT / "backups"
    backup_dir.mkdir(exist_ok=True)
    zip_path = backup_dir / f"TrashNet_E3_DirectML_{datetime.now():%Y%m%d_%H%M%S}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in source.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(source.parent))
    print(f"Backup criado: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
