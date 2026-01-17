# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-02.csv`
- Размер: (18000, 39)
- Целевая переменная: `target` (классы 0, 1 в пропорциях ~0.74, ~0.26)
- Признаки: 37 признаков типа float64 (2 столбца int64 (id и target) исключены)

## 2. Protocol

- Разбиение: train/test (0.75/0.25, `random_state` = 42) со стратификацией по целевому признаку
- Подбор: CV на train (5-фолдовая стратифицированная кросс-валидация на train (StratifiedKFold, shuffle=True, random_state=42))
- Оптимизировали: ROC-AUC (Area Under ROC Curve)
- Метрики: accuracy, F1, ROC-AUC
    - ROC-AUC: Основная метрика для оптимизации, хорошо работает с несбалансированными данными, оценивает качество ранжирования вероятностей
    - F1: Балансирует precision и recall, важна при несбалансированных классах
    - Accuracy: Базовая метрика для общего понимания доли правильных ответов

## 3. Models

- DummyClassifier (baseline): Стратегия "most_frequent" (предсказывает самый частый класс 0)
- LogisticRegression (baseline из S05):
    - Pipeline со StandardScaler
    - Гиперпараметры: C = [0.1, 1.0, 10.0], penalty = ["l2"], solver = ["lbfgs"]
    - Лучшие параметры: C = 1.0
- DecisionTreeClassifier:
    - Контроль сложности через: max_depth = [None, 3, 5, 8], min_samples_leaf = [1, 5, 10, 20], ccp_alpha = [0.0, 0.001, 0.005, 0.01]
    - Лучшие параметры: ccp_alpha=0.0, max_depth=8, min_samples_leaf=20
- RandomForestClassifier:
    - n_estimators = 600 (фиксировано)
    - Гиперпараметры: max_depth = [None, 6, 10], min_samples_leaf = [1, 5, 10], max_features = ["sqrt", 0.5]
    - Лучшие параметры: max_depth=None, max_features='sqrt', min_samples_leaf=1
- HistGradientBoostingClassifier (boosting):
    - early_stopping = True
    - Гиперпараметры: learning_rate = [0.03, 0.05, 0.1], max_depth = [2, 3, None], max_leaf_nodes = [15, 31, 63]
    - Лучшие параметры: learning_rate=0.05, max_depth=None, max_leaf_nodes=63
- StackingClassifier (с CV-логикой):
    - Базовые модели: лучшие LogReg(scaled), RandomForest, HistGradientBoosting
    - Мета-модель: LogisticRegression
    - Внутренняя CV: 5 фолдов

## 4. Results

- Таблица/список финальных метрик на test по всем моделям:
    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
    <thead>
        <tr style="text-align: right;">
        <th></th>
        <th>accuracy</th>
        <th>f1</th>
        <th>roc_auc</th>
        <th>model</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <th>5</th>
        <td>0.916444</td>
        <td>0.831390</td>
        <td>0.935006</td>
        <td>Stacking</td>
        </tr>
        <tr>
        <th>4</th>
        <td>0.905778</td>
        <td>0.802791</td>
        <td>0.932145</td>
        <td>HistGradientBoosting</td>
        </tr>
        <tr>
        <th>3</th>
        <td>0.893556</td>
        <td>0.762754</td>
        <td>0.930085</td>
        <td>RandomForest</td>
        </tr>
        <tr>
        <th>2</th>
        <td>0.818667</td>
        <td>0.619048</td>
        <td>0.831701</td>
        <td>DecisionTree</td>
        </tr>
        <tr>
        <th>1</th>
        <td>0.816222</td>
        <td>0.571724</td>
        <td>0.800894</td>
        <td>LogReg(scaled)</td>
        </tr>
        <tr>
        <th>0</th>
        <td>0.737333</td>
        <td>0.000000</td>
        <td>0.500000</td>
        <td>Dummy(most_frequent)</td>
        </tr>
    </tbody>
    </table>
    </div>

- Победитель: StackingClassifier с ROC-AUC = 0.9350
 Stacking показал наилучшее качество, объединяя сильные стороны трех различных моделей (линейной, bagging и boosting). Ансамблевый подход позволил снизить дисперсию и получить более стабильные предсказания.

## 5. Analysis

- Устойчивость: 
    Значения метрик (accuracy, f1, roc_auc) для каждой модели не демонстрируют систематического роста или падения с увеличением random_state (0 → 20 → 60 → 80 → 100). Они колеблются в небольшом диапазоне вокруг своего среднего значения. Этот небольшой разброс означает, что качество модели не зависит от случайного стечения обстоятельств при разбиении данных на обучающую и тестовую выборки.
        <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
    <thead>
        <tr style="text-align: right;">
        <th>random_state</th>
        <th>accuracy</th>
        <th>f1</th>
        <th>roc_auc</th>
        <th>model</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <th>0</th>
        <td>0.907778</td>
        <td>0.805073</td>
        <td>0.936192</td>
        <td>HistGradientBoosting</td>
        </tr>
        <tr>
        <th>80</th>
        <td>0.914667</td>
        <td>0.824818</td>
        <td>0.934837</td>
        <td>HistGradientBoosting</td>
        </tr>
        <tr>
        <th>20</th>
        <td>0.906889</td>
        <td>0.803747</td>
        <td>0.933945</td>
        <td>HistGradientBoosting</td>
        </tr>
        <tr>
        <th>60</th>
        <td>0.909778</td>
        <td>0.813247</td>
        <td>0.930504</td>
        <td>HistGradientBoosting</td>
        </tr>
        <tr>
        <th>100</th>
        <td>0.903778</td>
        <td>0.799072</td>
        <td>0.927289</td>
        <td>HistGradientBoosting</td>
        </tr>
        <tr>
        <th>100</th>
        <td>0.828667</td>
        <td>0.659000</td>
        <td>0.843611</td>
        <td>DecisionTree</td>
        </tr>
        <tr>
        <th>0</th>
        <td>0.820000</td>
        <td>0.638393</td>
        <td>0.842237</td>
        <td>DecisionTree</td>
        </tr>
        <tr>
        <th>60</th>
        <td>0.829556</td>
        <td>0.653098</td>
        <td>0.841593</td>
        <td>DecisionTree</td>
        </tr>
        <tr>
        <th>20</th>
        <td>0.825111</td>
        <td>0.633442</td>
        <td>0.831515</td>
        <td>DecisionTree</td>
        </tr>
        <tr>
        <th>80</th>
        <td>0.834667</td>
        <td>0.665768</td>
        <td>0.836929</td>
        <td>DecisionTree</td>
        </tr>
        <tr>
        <th>80</th>
        <td>0.824000</td>
        <td>0.581837</td>
        <td>0.816315</td>
        <td>LogReg(scaled)</td>
        </tr>
        <tr>
        <th>0</th>
        <td>0.813556</td>
        <td>0.563703</td>
        <td>0.808872</td>
        <td>LogReg(scaled)</td>
        </tr>
        <tr>
        <th>60</th>
        <td>0.815778</td>
        <td>0.574654</td>
        <td>0.808375</td>
        <td>LogReg(scaled)</td>
        </tr>
        <tr>
        <th>20</th>
        <td>0.814222</td>
        <td>0.559536</td>
        <td>0.808167</td>
        <td>LogReg(scaled)</td>
        </tr>
        <tr>
        <th>100</th>
        <td>0.815556</td>
        <td>0.560847</td>
        <td>0.811048</td>
        <td>LogReg(scaled)</td>
        </tr>
    </tbody>
    </table>
    </div>

- Ошибки: confusion matrix для лучшей модели + комментарий

    Матрица ошибок:

    | Факт \ Прогноз | Предсказан класс 0 | Предсказан класс 1 |
    |----------------|-------------------|-------------------|
    | **Реальный класс 0** | 3197 (TN) | 121 (FP) |
    | **Реальный класс 1** | 255 (FN) | 927 (TP) |

    Качество предсказания для большинства класса (0) — отличное.
    Для класса 1 же модель демонстрирует низкую полноту, однако высокую точность предсказания. Это следствие дисбаланса классов.
- Интерпретация: permutation importance (top-10/15) + выводы

    Признак f16 является наиболее важным для модели со значением mean importance (roc_auc drop) равным 0.06. Метрика roc_auc изменяется в диапазоне [0.5, 1.0], поэтому максимальное падение всего 0.06 означает, что модель НЕ опирается на один-два ключевых признака. Это признак хорошей, робастной модели — она не "переобучается" на шум в одном признаке.

## 6. Conclusion

Интерпретируемость — главное преимущество деревье решений: можно проследить путь от корня к листу и понять логику предсказания. Однако без ограничений они создают слишком сложные деревья, которые идеально запоминают обучающие данные, но плохо обобщают.

Ансамбли объединяют сотни деревьев, компенсируя ошибки отдельных моделей (снижают дисперсию), что дает выигрыш в точности за счёт потери простоты интерпретации (черный ящик).

Random Forest — параллельное обучение независимых деревьев на случайных подвыборках (bagging).
Gradient Boosting — последовательное обучение, где каждое новое дерево исправляет ошибки предыдущих.