<div align="center">

# Приоритизация обращений

**Решение тестового задания Avito**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-ML-00A2FF)](https://catboost.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperopt-2C3E50)](https://optuna.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458)](https://pandas.pydata.org/)

</div>

<p align="center">
  <strong>Daily AP на публичном тесте: <span style="color:#00C853">0.70791</span></strong>
</p>

---

## О задаче

Задача — оценить вероятность успешного целевого действия в течение 5 дней после назначения обращения. Модель используется для ранжирования: более перспективные обращения должны получать более высокий `score`.

**Основная метрика** — **Daily Average Precision** (усредненный AP по дням назначения): 

Пусть D - множество дат назначения обращений в скрытой тестовой выборке. Для каждой даты считаем Average Precision только по обращениям, назначенным в этот день:
$$AP_{d} = AP({(y_i, s_i): assignment date_i = d})$$

Итоговая метрика - среднее значение по дням:
$$DailyAP = \frac{1}{|D|}\sum_{d \in D}AP_d$$


---

## Что сделано

- **Feature Engineering** из `events.csv` с жёстким временным фильтром `event_ts < assignment_ts` (без утечки)
- Многооконные счётчики событий (12h, 1d, 3d, 7d, 14d, 30d)
- Recency по каждому типу событи
- Price статистики и price delta
- Last context (`last_ctx_seq`, `last_src_slot`)
- Feature selection по важности (CatBoost importance ≥ 0.05)
- Time-based expanding window CV (4 фолда)
- Ансамбль: **CatBoost + LightGBM + XGBoost** с равными весами
- Подбор гиперпараметров через **Optuna** (100 trials) с `MedianPruner`
- Финальный бленд с несколькими `random_state`

**Библиотеки**: `pandas`, `catboost`, `lightgbm`, `xgboost`, `optuna` — все open-source, локально.

---

## Результаты

| Этап                        | CV Daily AP       | Public Daily AP |
|----------------------------|-------------------|-----------------|
| Базовая версия             | ~0.634            | 0.662           |
| + улучшенные FE            | 0.649             | 0.6687          |
| + Optuna + feature select  | **0.669**         | **0.70791**     |

---

## Структура решения

```text
solution/
├── solution.ipynb          - основной ноутбук
├── submission.csv          - финальный файл
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── events.csv
└── README.md