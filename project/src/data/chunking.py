from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.data.preprocessing import (
    remove_update_timestamps,
    is_header_only_chunk,
)


def split_documents(
    documents: List[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
) -> List[Document]:

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []

    for doc in documents:

        text = remove_update_timestamps(doc.page_content)

        md_chunks = markdown_splitter.split_text(text)

        for md_chunk in md_chunks:
            combined_metadata = doc.metadata.copy()
            combined_metadata.update(md_chunk.metadata)
            md_chunk.metadata = combined_metadata

        md_chunks = [
            chunk
            for chunk in md_chunks
            if not is_header_only_chunk(chunk)
        ]

        final_chunks = []

        for chunk in md_chunks:

            if len(chunk.page_content) > chunk_size:

                sub_chunks = recursive_splitter.create_documents(
                    texts=[chunk.page_content],
                    metadatas=[chunk.metadata],
                )

                final_chunks.extend(sub_chunks)

            else:
                final_chunks.append(chunk)

        all_chunks.extend(final_chunks)
    
    for idx, chunk in enumerate(all_chunks):
      chunk.metadata["chunk_id"] = idx

    return all_chunks