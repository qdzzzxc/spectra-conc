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
## mendeley_raman — Raman Scattering gathering from pure glucose liquid (отклонён, см. docs/DATASETS.md)
- DOI: 10.17632/yk75sxk6p5.1, версия 1 (2024-02-29), лицензия CC BY 4.0.
- Скачано: 2026-11-12, `https://data.mendeley.com/public-files/datasets/yk75sxk6p5/files/<id>/file_downloaded`.

| Файл | id | Размер, байт | sha256 |
|---|---|---|---|
| `Fourteenlabels_RamanData.csv` | ef49316b-b728-479f-8a32-aea101e0f3d1 | 769775 | `663595467f41648904c340adff4d6deb55e2ce445d46a76cdf82d703cd574119` |
| `Fourteenlabels_labels.csv` | 8cb9260a-a8ab-4952-8808-0eee871c927d | 530 | `9aaa50f505352d24a76ed24ff20cb894d72d6e357c6982894dcd1fc6179f67cf` |
| `Tenlabels_RamanData.csv` | 3a8b0f88-8af6-4d8d-b2c9-17cde4ffb932 | 562631 | `e758c2ab74fa61ee0e7bb0e9c3392fdbd94096c31b1111e652e712d34146b2c6` |
| `Tenlabels_labels.csv` | 89f2184c-7f9e-40e1-910c-6f47c81db5e8 | 390 | `5aba09ea4f1e220b992bca442cc86e7798b2a946f559e1effae13d3f5724a697` |

Описание содержимого — `docs/DATASETS.md`.

## cambridge — Chip-scale sensor for spectroscopic metrology

- [DOI 10.17863/CAM.110905](https://doi.org/10.17863/CAM.110905), CC BY 4.0.
- Скачано 2026-11-23. Оригинальный Excel сохранён как `raw/cambridge/glucose.xlsx`.
- SHA-256 `8493a9028634eec731f4f41eb1580b2b58d9cce4274bcc13ea8b0587e1db1638`.

## lancaster — водные растворы глюкозы и интралипид

- [DOI 10.17635/lancaster/researchdata/666](https://doi.org/10.17635/lancaster/researchdata/666), CC BY.
- Скачано вручную 2026-11-23. Оригинал сохранён как `raw/lancaster/Intralipid_and_aq_glucose_comparison.opju`.
- SHA-256 `2e6027f985c6fecf6f8e539dff5cd55d875a40ee8f2ffdf746706940cd592430`.
- `opj2dat` из liborigin 3.0.0 не извлёк таблицы. Для доступа к ним можно открыть проект в [Origin Viewer](https://www.originlab.com/viewer/) и скопировать листы или сохранить проект в формате OPJ.
- Экспорт из Windows сохранён в `raw/lancaster/origin_export/`. Манифест содержит 49 листов. Размеры CSV и таблиц метаданных сверены с манифестом. SHA-256 архива экспорта `1dc729e078427c8a70cb612d236b1e02910219e90debf0e16c900e249446d930`.
