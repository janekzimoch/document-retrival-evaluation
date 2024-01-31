import os
import sys
sys.path.append(os.getcwd())
import json
from dotenv import load_dotenv
from tqdm import tqdm

from llama_index.embeddings import HuggingFaceEmbedding
from sklearn.metrics.pairwise import cosine_similarity

import evaluation.metrics as metrics
import evaluation.groundtruth_relevance as gt_util
from embedding_store import EmbeddingStore

load_dotenv()

METRICS = {'recall': [5, 20], 'r_precision': [None], 'ndcg': [0.1], 'ndcg_exp': [0.1], 'mrr': [10], 'map': [None]}


def load_dataset():
    with open(os.environ.get('DORIS_MAE_DATASET_DIR'), "r") as f:
        dataset = json.load(f)
    return dataset

def load_model_results():
    ''' data format, list of ordered lists: [[top_1, top_2, ...], ... ] '''
    with open(os.environ.get('MODEL_EVAL_OUTPUT'), "r") as f:
        dataset = json.load(f)
    return dataset

def get_template_ranking(dataset):
    ''' ranking in the format expected by functions from the original repo with the dataset. 
    expected format: [{'index_rank': [top_1, top_2,...]}, {...}] '''
    template_ranking = [{'index_rank': query['candidate_pool']} for query in dataset['Query']]
    return template_ranking

def get_embeddings(texts, embedding_model):
    embeddings = embedding_model.embed_batch(texts)
    return embeddings

def get_distances(embedding, candidate_pool_embeddings):
    distances = cosine_similarity([embedding], candidate_pool_embeddings)[0]
    assert len(distances) == len(candidate_pool_embeddings)
    return distances

def rank_candidate_pools(distances_all):
    ''' takes `distances_all` dictionary (i.e. {'a': 2, 'b': 0, 'c': 1}) 
    and returns list of sorted keys by descending order of their values (i.e. ['a', 'c', 'b'])
    '''
    rank = []
    for distances in distances_all:
        ranked_indexes = [key for key, value in sorted(distances.items(), key=lambda item: item[1], reverse=True)]
        rank.append({'index_rank': ranked_indexes})
    return rank

def get_model_ranking(dataset, model_name):
    rank = []
    embedding_model = EmbeddingStore(model_name, dataset, 10, initialise=False)
    print(f'\n {model_name}')

    # convert queries to embeddings
    texts = [query['query_text'] for query in dataset["Query"]]
    embeddings = get_embeddings(texts, embedding_model)

    # extract embeddingsd of articles from the candidate pool - {article_id: embedding}
    candidate_pool_embeddings = []
    candidate_pools = [query['candidate_pool'] for query in dataset['Query']]
    print('generating embeddings...')
    for candidate_pool in tqdm(candidate_pools):
        abstract_texts = [dataset['Corpus'][idx]['original_abstract'] for idx in candidate_pool]
        abstract_embeddings = get_embeddings(abstract_texts, embedding_model)
        candidate_pool_embeddings.append(abstract_embeddings)
    assert len(candidate_pool_embeddings) == len(embeddings)

    # score distance to the query - {article_id: distance}
    distances_all = []
    print('getting distances...')
    for i in tqdm(range(len(embeddings))):
        distances = get_distances(embeddings[i], candidate_pool_embeddings[i])
        distances_dict = {idx: distance for idx, distance in zip(candidate_pools[i], distances)}
        distances_all.append(distances_dict)

    # sort from lowest to highest distance
    rank = rank_candidate_pools(distances_all)
    return rank

def print_model_results(gt_scores, ranking):
    ' to be modified to the same format as random model. Preferably JSON format. '
    for test_name in METRICS.keys():
        for k in METRICS[test_name]:
            value = metrics.get_metric(gt_scores, ranking, test_name, k)
            if k != None:
                print(f"{test_name}@{k}  : {value}%")
            else:
                print(f"{test_name}  : {value}%")


if __name__ == '__main__':
    dataset = load_dataset()
    
    ground_truth_scores = gt_util.compute_ground_truth_scores(dataset)
    ground_truth_ranking = gt_util.order_results_by_score(ground_truth_scores)
    
    template_ranking = get_template_ranking(dataset)
    random_ranking_results = metrics.get_baseline_performance(ground_truth_scores, template_ranking, METRICS, trials = 100)
    metrics.print_results(METRICS, random_ranking_results, 'random')

    base_model = "BAAI/bge-base-en-v1.5"
    large_model = "BAAI/bge-large-en-v1.5"
    
    base_model_ranking = get_model_ranking(dataset, base_model)
    base_model_ranking_results = {}
    metrics.get_performance_eval(ground_truth_scores, base_model_ranking, base_model_ranking_results, METRICS)
    metrics.print_results(METRICS, base_model_ranking_results, base_model)

    large_model_ranking = get_model_ranking(dataset, large_model)
    large_model_ranking_results = {}
    metrics.get_performance_eval(ground_truth_scores, large_model_ranking, large_model_ranking_results, METRICS)
    metrics.print_results(METRICS, large_model_ranking_results, large_model)