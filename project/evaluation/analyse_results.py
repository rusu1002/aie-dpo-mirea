import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.config import settings


def generate_evaluation_report():

    Path(settings.EVAL_REPORT_DIR).mkdir(parents=True, exist_ok=True)
    with open(settings.EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for row in data:
        records.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "faithfulness": row["faithfulness"]["score"],
                "answer_relevancy": row["answer_relevancy"]["score"],
                "contextual_relevancy": row["contextual_relevancy"]["score"],
            }
        )

    df = pd.DataFrame(records)

    # ==================================
    # СТАТИСТИКА
    # ==================================

    def calc_stats(series):
        return {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),

            "below_05": int((series < 0.5).sum()),
            "below_07": int((series < 0.7).sum()),
            "above_09": int((series >= 0.9).sum()),
        }


    statistics = {metric: calc_stats(df[metric]) for metric in ["faithfulness", "answer_relevancy", "contextual_relevancy"]}

    with open(Path(settings.EVAL_REPORT_DIR)/"statistics.json", "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

    # ==================================
    # КОРРЕЛЯЦИЯ
    # ==================================

    corr = df[["faithfulness", "answer_relevancy", "contextual_relevancy"]].corr()

    # corr.to_csv(Path(settings.EVAL_REPORT_DIR)/"correlation_matrix.csv")

    plt.figure(figsize=(6, 5))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=30)
    plt.yticks(range(len(corr.columns)), corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()

    plt.savefig(Path(settings.EVAL_REPORT_DIR)/"correlation_heatmap.png", dpi=300)

    plt.close()

    # ==================================
    # ГИСТОГРАММЫ
    # ==================================

    for metric in ["faithfulness", "answer_relevancy", "contextual_relevancy"]:
        plt.figure(figsize=(8, 5))
        plt.hist(df[metric], bins=20)
        plt.title(metric)
        plt.xlabel("score")
        plt.ylabel("count")

        plt.tight_layout()

        plt.savefig(Path(settings.EVAL_REPORT_DIR)/f"{metric}_histogram.png", dpi=300)

        plt.close()

    # ==================================
    # ХУДШИЕ СЛУЧАИ
    # ==================================

    worst_cases = {}

    for metric in ["faithfulness", "answer_relevancy", "contextual_relevancy"]:
        worst_cases[metric] = (df.sort_values(metric).head(10).to_dict(orient="records"))

    with open(Path(settings.EVAL_REPORT_DIR)/"worst_cases.json", "w", encoding="utf-8") as f:
        json.dump(worst_cases, f, ensure_ascii=False, indent=2)

    # ==================================
    # ЛУЧШИЕ СЛУЧАИ
    # ==================================

    best_cases = {}

    for metric in ["faithfulness", "answer_relevancy", "contextual_relevancy"]:
        best_cases[metric] = (df.sort_values(metric, ascending=False).head(10).to_dict(orient="records"))

    with open(Path(settings.EVAL_REPORT_DIR)/"best_cases.json", "w", encoding="utf-8") as f:
        json.dump(best_cases, f, ensure_ascii=False, indent=2)

    # ==================================
    # ИТОГОВЫЙ ОТЧЕТ
    # ==================================

    summary = {
        "num_questions": len(df),
        "average_faithfulness": float(df["faithfulness"].mean()),
        "average_answer_relevancy": float(df["answer_relevancy"].mean()),
        "average_contextual_relevancy": float(df["contextual_relevancy"].mean()),
    }
    with open(Path(settings.EVAL_REPORT_DIR)/"summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

# def main():
#     generate_evaluation_report()

# if __name__ == "__main__":
#     main()