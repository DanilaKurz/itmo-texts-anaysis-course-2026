# Данила Курз — решения лабораторных работ

Курс «Анализ текстов», 2026. Датасет — Lenta.ru News Dataset.

## Содержимое

| Папка | Лабораторная | Статус |
|---|---|---|
| [`lab1/`](lab1/README.md) | Очистка текста, стемминг/лемматизация, расстояние Левенштейна, n-граммная модель | ✅ готово |
| [`lab2/`](lab2/README.md) | POS-разметка, деревья зависимостей, извлечение паттернов, разрешение анафоры | ✅ готово |
| `lab3/` | Векторизация, классификация, BM25, LDA | в работе |
| `lab4/` | Word2Vec, BiLSTM, fine-tuning BERT, бенчмарк | в работе |
| [`src/textkit/`](src/textkit/) | Переиспользуемые модули: загрузка датасета, очистка, нормализация, Левенштейн, языковая модель, морфология | — |

Каждая лабораторная — это `run.py` (воспроизводимый сценарий) + `README.md`
(отчёт с числами) + `artifacts/` (графики, метрики, логи). Все числа в отчётах
получены запуском `run.py` на реальных данных; повторный запуск даёт те же
значения.

## Как воспроизвести

```bash
# 1. Python 3.12 + зависимости
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m spacy download ru_core_news_sm

# 2. Датасет (в data/lenta-ru-news, ~860 МБ; не коммитится в git)
mkdir -p data/lenta-ru-news
curl -L -o data/lenta-ru-news/train.parquet \
  "https://huggingface.co/datasets/zloelias/lenta-ru/resolve/main/data/train-00000-of-00001.parquet"
curl -L -o data/lenta-ru-news/test.parquet \
  "https://huggingface.co/datasets/zloelias/lenta-ru/resolve/main/data/test-00000-of-00001.parquet"

# 3. Прогон
python lab1/run.py    # ~6 минут
python lab2/run.py    # ~25 минут (POS-статистика и парсинг кешируются)
```

Замеры на Apple M5 (10 ядер, 16 ГБ): лаба 1 — ~6 минут (очистка кешируется),
лаба 2 — POS-разметка 50 тыс. документов ~5 минут, парсинг 20 тыс. документов
~15–20 минут (один процесс: многопоточный `n_process` в spaCy на macOS
замедляет разбор в 2.5 раза из-за spawn-старта воркеров).

## Параметры запуска

| Константа | Файл | Значение | Зачем |
|---|---|---|---|
| `test_size=0.2, stratify` | `lab1/run.py` | сплит 80/20 | единый сплит для всех лаб |
| `POS_SAMPLE=50000` | `lab2/run.py` | POS-подвыборка | доли POS стабилизируются раньше полного корпуса |
| `PARSE_SAMPLE=20000` | `lab2/run.py` | подвыборка для парсинга | полный корпус не нужен для 10 деревьев и топ-20 пар |
