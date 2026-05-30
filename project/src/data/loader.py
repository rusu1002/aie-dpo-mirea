from typing import List
import ast

import pandas as pd
from langchain_core.documents import Document


def load_docs(file_path: str) -> List[Document]:
    """
    Загружает документы из CSV-файла.

    Ожидаемые колонки:
    - id
    - text
    - tags
    """

    df = pd.read_csv(file_path)

    documents = []

    for _, row in df.iterrows():

        tags = row.get("tags", [])

        if isinstance(tags, str):
            try:
                tags = ast.literal_eval(tags)
            except Exception:
                tags = [tags]

        metadata = {
            "id": row["id"],
            "tags": tags,
        }

        documents.append(
            Document(
                page_content=row["text"],
                metadata=metadata
            )
        )

    return documents