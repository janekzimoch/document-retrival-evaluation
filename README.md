# Quantitative Evaluation of Embedding Vectors for Scientific Papers

## Purpose

This repository is dedicated to the quantitative evaluation of embedding vectors' ability to represent the semantics of scientific papers in the field of Machine Learning, using the Doris-Mae dataset. Establishing a well-grounded evaluation framework is crucial for the development of machine learning-enabled applications. This codebase guides the development of embedding vectors for information retrieval in my other projects.

## Dataset

The dataset was sourced from a paper by Wang et. al., which shares a dataset for performance evaluation of document retrieval systems. Their work aligns closely with my research project aimed at building an AI search engine for machine learning papers.

- **Paper**: [Performance Evaluation of Document Retrieval Systems](https://arxiv.org/pdf/2310.04678.pdf)
- **GitHub Repository**: [Doris-Mae Dataset](https://github.com/Real-Doris-Mae/Doris-Mae-Dataset/tree/main)
- **Dataset Download**: [Zenodo Record](https://zenodo.org/records/8299749)

## Results

The objective was to identify which textual-embedding-vector model is most suitable for information retrieval from scientific papers in the field of ML. The models `BAAI/bge-base-en-v1.5` and `BAAI/bge-large-en-v1.5` were tested against the OpenAI Ada model and a random ranking baseline.

| Model                 | R@5  | R@20 | RP   | NDCG exp 10% | MRR@10 | MAP  |
|-----------------------|------|------|------|--------------|--------|------|
| Random                | 4.60 | 18.53| 16.41| 7.12         | 3.37   | 19.67|
| Ada                   | 15.38| 42.84| 35.81| 27.46        | 19.88  | 40.37|
| BAAI/bge-base-en-v1.5 | 18.40| 44.26| 37.99| 27.84        | 17.87  | 42.19|
| BAAI/bge-large-en-v1.5| 17.36| 47.24| 37.34| 27.44        | 14.99  | 41.23|

## Notes

Running large embedding vector models is resource-intensive, requiring embedding of hundreds of thousands of documents. Utilizing a GPU can significantly speed up the process. Resources for accessing GPU-equipped Virtual Machines include:

- [Google Cloud - $300 free credits](https://cloud.google.com/free?hl=en)
- [Microsoft Azure - $200 initial credit](https://azure.microsoft.com/en-us/pricing/offers/ms-azr-0044p)

For my experiments, I utilized Google Cloud and created a virtual machine from [this pre-configured image](https://cloud.google.com/deep-learning-vm), which includes Python and CUDA drivers.

## Code Structure

- `main.py` - The main script for launching and orchestrating other components.
- `embedding_store.py` - Manages data ingestion (e.g., from JSON, CSV, or PDF), embedding text into vectors, setting up FAISS vector store, and retrieval functionality.
- `evaluation/evaluate.py` - Orchestrates the evaluation of models, loading & running models, and printing metrics to the command line.
