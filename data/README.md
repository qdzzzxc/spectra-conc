# data/
`raw/` — скачанные файлы (не коммитятся). Для каждого датасета: папка `raw/<name>/` + запись здесь: ссылка/DOI, дата скачивания, версия, лицензия.

## ntnu — Aqueous glucose measured by NIR spectroscopy
- DOI: 10.18710/NSHFAK (DataverseNO), версия 2.1 (релиз 2023-09-28), лицензия CC0 1.0.
- Скачано: 2026-11-12, через API: `https://dataverse.no/api/access/datafile/<id>?format=original`.

| Файл | id | Размер, байт | sha256 |
|---|---|---|---|
| `00_Readme.txt` | 113890 | 3124 | `7e7c939d8be24a9cd05cf5c1b869405e04df32679055cfc0ea218e2be1f680f7` |
| `CalibrationData.txt` | 86652 | 48515400 | `d1b8b01b215d8b20bd67c1a4254c05f04542ba14645443efb5d586371c250269` |
| `ValidationData.txt` | 86651 | 18377328 | `e7c8d5777febb88f38a40241ed5af98d1611b7cab72139287b194391c2cf95c4` |

Повторить:
```bash
mkdir -p data/raw/ntnu && cd data/raw/ntnu
for p in "113890 00_Readme.txt" "86652 CalibrationData.txt" "86651 ValidationData.txt"; do
  set -- $p; curl -sSL "https://dataverse.no/api/access/datafile/$1?format=original" -o "$2"; done
sha256sum *
```
Описание содержимого — `docs/DATASETS.md`.
