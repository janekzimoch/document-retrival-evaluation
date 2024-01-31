import os
from collections import deque
from enum import Enum

import fitz
from interfaces import Document
import utils.utils as utils


class ChunkingMethods(str, Enum):
    page = "page"
    paragraph = "paragraph"
    chunk = "chunk"


class ParserPDF():

    def __init__(self, metadata_path):
        self.metadata = utils.readJson(metadata_path)

    def _is_pdf(self, path: str) -> None:
        if not path.endswith('.pdf'):
            raise TypeError(f"{path} - is not a PDF")

    def _is_file(self, path: str) -> None:
        if not os.path.isfile(path):
            raise TypeError(f"{path} - is not a file")

    def _is_directory(self, path: str) -> None:
        if not os.path.isdir(path):
            raise TypeError(f"{path} - is not a directory")

    def extract_file(self, file_path: str, chunking_method: ChunkingMethods = ChunkingMethods.page) -> Document:
        doc_texts = []
        doc_title = file_path.split('/')[-1].split('.pdf')[0]
        self._is_file(file_path)
        self._is_pdf(file_path)

        if chunking_method is ChunkingMethods.paragraph:
            raise NotImplementedError(
                f"{chunking_method} - this chunking method has not yet been implemented")

        if chunking_method is ChunkingMethods.chunk:
            raise NotImplementedError(
                f"{chunking_method} - this chunking method has not yet been implemented")

        if chunking_method is ChunkingMethods.page:
            doc = fitz.open(file_path)
            N = len(doc)
            for n in range(N):
                page = doc[n]
                words = page.get_text("words")
                words = [element[4] for element in words]
                page_text = ' '.join(words)
                doc_texts.append(page_text)
        else:
            raise KeyError(
                f"{chunking_method} - is an invalid chunking methods. Use one of the available: {ChunkingMethods}")

        document = {**self.metadata[doc_title], 'text': doc_texts}
        return document

    def extract_directory(self, dir_path: str) -> list[Document]:
        self._is_directory(dir_path)

        docs = []
        paths = deque([dir_path])
        while paths:
            current_path = paths.popleft()

            if os.path.isdir(current_path):
                new_paths = os.listdir(current_path)
                paths.extend([os.path.join(current_path, path)
                             for path in new_paths])

            if os.path.isfile(current_path):
                try:
                    doc_texts = self.extract_file(current_path)
                    docs.append(doc_texts)
                except Exception as e:
                    print(e)
                    continue
        return docs


if __name__ == "__main__":
    parser = ParserPDF()
    docs = parser.extract_directory(
        '/Users/janek/Documents/mlreference/papers/test')
    print(docs)
