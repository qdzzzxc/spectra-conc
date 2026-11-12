# Git workflow

## Branches

- Использовать две постоянные ветки: main и test.
- Изменять main через merge из test или промежуточной ветки, явно предусмотренной правилами repository.
- Вносить изменения в test только через merge request.
- Называть короткоживущие ветки feat/<description> или fix/<description>.
- Не смешивать независимые fixes/features в одной ветке.

## Commit messages

Формат:

~~~text
<type>[optional scope]: <краткое описание>

[optional body]

[optional footer]
~~~

Использовать типы feat, fix, refactor, docs, style, test, chore, perf или wip. После type/scope ставить двоеточие и пробел.

- Писать краткое описание лаконично и по фактическому diff.
- Писать описание на русском, если repository precedent не требует другого.
- Добавлять body после одной пустой строки, когда нужно объяснить мотив или ограничения.
- Добавлять issue link и migration metadata в footer.
- Не добавлять Co-authored-by.
- Для несовместимости начинать body или footer точной строкой BREAKING CHANGE: и объяснять, что сломано и как мигрировать.

## Merge requests

- Один MR должен закрывать один fix или feature и не затрагивать лишнюю логику.
- Добавлять краткое описание изменений, способ проверки и важные риски.
- Не считать зелёный CI достаточным доказательством готовности: проверить diff, тестовые результаты и rollout risk.
- Не выполнять merge, если пользователь явно не включил его в задачу.

## Stacked MRs

При diff больше примерно 500 строк разбивать feature на зависимые MR по логическим слоям, например:

1. types и models;
2. infrastructure;
3. business logic;
4. entrypoint/integration.

Каждый следующий MR ветвить от предыдущего. В описании каждого MR указывать:

- полный функционал и часть текущего MR;
- parent и dependent MRs;
- порядок зависимостей и merge;
- влияние rebase нижнего слоя на верхние.

Не force-push shared stack без явного разрешения пользователя.

## Repository conventions

- Размещать repository в подходящей subgroup: Agents, LLM, RAG, SOC Automations, Templates, Infrastructure, Descriptions Generation, Classifiers, Libraries, Data Engineering, Test Deployment, Junk или TI.
- Называть subgroup с заглавных букв через пробел.
- Называть repository в kebab-case.
- Использовать lower snake_case в ссылках и путях, если внутренний convention требует этого.
- Добавлять README каждому repository.
- Для library описывать модули и API; примеры можно дать notebook.
- Для service описывать структуру, основные зависимости и API contracts.

## Git LFS

Использовать Git LFS для model weights, datasets, archives, images и других крупных binaries. Добавлять patterns в .gitattributes и коммитить этот файл.

~~~text
git lfs install
git lfs track "*.pt" "*.onnx" "*.bin"
git add .gitattributes
~~~

Проверять, не попал ли binary в обычную Git history до добавления LFS. Не переписывать историю без явного запроса пользователя и проверки точных затрагиваемых refs.
