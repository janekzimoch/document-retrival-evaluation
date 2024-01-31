''' file for generating ranking results in bulk
At the moment not abstracted and serving this dataset: 
DORIS-MAE_dataset_v1.json '''

from embedding_store import EmbeddingStore
import evaluation.evaluate as evaluator 
import torch

if __name__ == '__main__':
    torch.cuda.empty_cache()
    dataset = evaluator.load_dataset()

    base_model = "BAAI/bge-base-en-v1.5"
    large_model = "BAAI/bge-large-en-v1.5"
    
    # es_base = EmbeddingxStore(base_model, dataset, 12)
    es_large = EmbeddingStore(large_model, dataset, 10)