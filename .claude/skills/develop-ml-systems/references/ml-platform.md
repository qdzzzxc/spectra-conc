# ML/LLM platform baseline

## Internal components

Перед реализацией искать нужную возможность в зависимостях проекта, внутренних библиотеках и шаблонах:

| Компонент | Назначение |
|---|---|
| vaulty | Секреты HashiCorp Vault, Pydantic/FastAPI integration |
| toolbox | Логирование, timing, TOML metadata, OpenTelemetry, HTTP/LiteLLM context |
| s3client | Работа с S3-compatible MinIO: загрузка, скачивание и администрирование, включая удаление бакетов |
| ti-connectors | Threat Intelligence API |
| qb | Builder SQL-запросов; подключать по необходимости, когда в проекте много SQL |
| data-pipeline | ETL sources и transformations |
| ml-pipeline | Управление моделями ML, кастомные реализации loss-функций и алгоритмов |

Не писать локальный wrapper поверх Vault, S3, logging или HTTP, если компонент из этого baseline предоставляет требуемый contract.

Для очередей и фоновых задач использовать Celery с Redis или RabbitMQ, а не внутреннюю библиотеку очередей.

## LiteLLM

- Для сервиса использовать Virtual Key, созданный под Service Account и проект. Личный Virtual Key использовать только для личных задач.
- Не хранить Virtual Key в коде или открытом env-файле.
- Передавать обязательные теги:

  - Project: имя сервиса без пробелов и заглавных букв;
  - Version: SemVer сервиса;
  - Env: prod, test или local.

- Передавать теги списком в metadata.tags или заголовком x-litellm-tags.
- Формировать один session_id на логическую сессию: pipeline_name:date:uuid.
- Формировать уникальный request_id на каждый вызов: service_name:action:date:uuid4.
- Использовать request-scoped context из toolbox.litellm, если он доступен в проекте.
- Валидировать структурированный ответ модели до использования.
- Ограничивать timeout, retries и output size. Делать retry только для повторяемых безопасно ошибок.
- Не отправлять внутренние или персональные данные внешнему provider без явной разрешённой схемы.

## Data and evaluation

- Зафиксировать схему и происхождение входов, версию датасета, модель, prompt/config и threshold.
- Исключить leakage между train, validation и test, а также leakage будущего времени.
- Проверить train/serve skew: preprocessing, missing values, categories, tokenization и feature defaults.
- Сравнивать с простым baseline и предыдущей production-версией.
- Оценивать не только среднюю метрику, но и критические slices, ошибки и abstention/fallback.
- Версионировать артефакты и обеспечить воспроизводимый путь от результата к коду и данным.
- Для LLM/RAG включать representative golden cases, retrieval quality, grounding, citation validity, injection cases и budget/latency.

## Orchestration

- Новые DAG писать на Prefect и начинать с внутреннего prefect-template.
- Использовать @flow как точку входа, @task как наблюдаемую единицу работы и get_run_logger() для Prefect UI.
- Отключать task cache для чтения свежих данных, API calls, DB writes и Prefect Variable updates.
- Использовать Variable только для небольшого состояния между запусками: cursor, last_id, last_sync.
- Делать side effects идемпотентными до добавления retry.
- Для legacy Airflow не добавлять в DAG крупную бизнес-логику и сторонние библиотеки. Вынести код в отдельный репозиторий/образ.
- В legacy Airflow сохранять существующий формат owner. Если owner обязателен, но его значение не задано в repository или запросе, не придумывать личность: оставить явный placeholder и сообщить о неполной конфигурации. Добавлять docstring с описанием и ссылкой на проект, передавать doc_md=__doc__.

## Storage and repositories

- Для S3-compatible storage использовать s3client, не новый wrapper над boto/minio.
- Не хранить веса, датасеты, архивы и крупные изображения напрямую в Git; использовать Git LFS и коммитить .gitattributes.
- Для notebook очищать outputs и metadata перед коммитом. Не считать notebook единственным production entrypoint.

## Operational telemetry

- Логировать model name/version, service version, environment, latency, result status и request correlation без prompt/secret payload.
- Для inference использовать bounded metric labels: endpoint template, model version, status, verdict из конечного словаря.
- Не использовать customer ID, request ID, raw prompt, raw URL и exception message как metric label.
- Различать технические inference errors, invalid model output, low confidence и business-negative result.

## Verification questions

- Можно ли повторить результат из версии кода, данных, модели и конфигурации?
- Что произойдёт при пустом input, schema drift, timeout, partial batch и provider outage?
- Есть ли безопасный rollback и совместимый output contract?
- Достигает ли evaluation заданных метрик на доступных representative данных?
- Сохраняются ли privacy, cost, latency и observability при масштабировании?

Не объявлять production readiness без доступных результатов evaluation и runtime-проверок. Недоступные проверки перечислять как непроверенные, а не считать успешно пройденными.
