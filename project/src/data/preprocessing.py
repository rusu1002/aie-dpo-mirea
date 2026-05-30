import re

from langchain_core.documents import Document


def remove_update_timestamps(text: str) -> str:
    """
    Удаляет строки вида:
    Обновлено 12.01.2024 в 15:30
    """

    pattern = r"^\s*Обновлено \d{2}\.\d{2}\.\d{4} в \d{2}:\d{2}\s*$"

    cleaned_lines = [
        line
        for line in text.splitlines()
        if not re.fullmatch(pattern, line)
    ]

    return "\n".join(cleaned_lines)


def is_header_only_chunk(doc: Document) -> bool:
    """
    Проверяет, содержит ли чанк только markdown-заголовок.
    """

    content = doc.page_content.strip()

    if not content:
        return True

    if re.fullmatch(r"#{1,6}\s+.+", content):
        return True

    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    if len(lines) == 1 and re.match(r"#{1,6}\s+.+", lines[0]):
        return True

    return False