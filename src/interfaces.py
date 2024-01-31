from pydantic import BaseModel


class Document(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    text: list[str]
    pdf: str
    year: int
    conference: str
    keywords: list[str]


class Recommendation(BaseModel):
    document: Document
    relevance_description: str
