import os
import sys
import json
import gc
from dotenv import load_dotenv
load_dotenv()
import torch 

from langchain.vectorstores.faiss import FAISS
from tqdm import tqdm
# from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings import HuggingFaceEmbedding

from interfaces import Document, Recommendation
from parsing.parserPDF import ParserPDF
import miscelanous.data_formater as formater
import miscelanous.utils as utils





class EmbeddingStore():

    def __init__(self, model_name="BAAI/bge-base-en-v1.5", data = None, batch_size=32, initialise = True):
        self.embedding_model = HuggingFaceEmbedding(
            model_name=model_name,
            )
        
        self.model_name = model_name.split('/')[1]
        self.batch_size = batch_size
        
        print('Embedder device: ', self.embedding_model._device)
        if initialise:
            self.embeddings_path = os.environ.get('EMBEDDINGS_PATH') + f'/embeddings_{self.model_name}.csv'
            self.vector_store_path = os.environ.get('FAISS_PATH') + f'/{self.model_name}'
            self.vector_store = self._initialise_vectore_store(data)
        print('Setup finished!')


    def _initialise_vectore_store(self, data):
        if os.path.exists(self.vector_store_path):
            vector_store = FAISS.load_local(self.vector_store_path, self.embedding_model)
            return vector_store
        else:
            
            if data is None:
                # load default data
                documents_dir = os.environ.get('PDF_ARTICLES_DIR')
                metadata_path = os.environ.get('PDF_ARTICLES_METADATA_DIR')
                parser = ParserPDF(metadata_path)
                docs = parser.extract_directory(documents_dir)
            else:
                # we assume data is from DORIS_MAE daatset
                docs: list[Document] = formater.convert_DORIS_MAE_to_my_format(data)

            # initialise vectore store
            print('preparing vectorstore...')
            texts, metadata = self._prepare_documents_for_faiss(docs)
            if not os.path.exists(self.embeddings_path):
                torch.cuda.empty_cache()
                self.embed_batch(texts, self.embeddings_path)
            embeddings = utils.load_data_csv(self.embeddings_path)
            text_embedding_pairs = list(zip(texts, embeddings))
            
            vector_store = FAISS.from_embeddings(
                text_embedding_pairs, self.embedding_model, metadatas=metadata)
            
            utils.make_directory(self.vector_store_path)
            vector_store.save_local(self.vector_store_path)
            return vector_store

    def _prepare_documents_for_faiss(self, docs: list[Document]):
        ''' each Document is divided into chunks. We need to seperate each chunk into a seperate element of a list to prepare this as an input to FAISS. '''
        text_list = []
        metadata_list = []
        for doc in docs:
            metadata = {'authors': doc['authors'],
                        'conference': doc['conference'],
                        'year': doc['year'],
                        'abstract': doc['abstract'],
                        'pdf': doc['pdf'],
                        'title': doc['title']}
            for text in doc['text']:
                text_list.append(text)
                metadata_list.append(metadata)
        return text_list, metadata_list

    def article_recommendations(self, query: str, K: int = 5) -> list[Recommendation]:
        most_similar = self.get_most_similar(query, K)
        recommendations = []
        for document, score in most_similar:
            relevance_description = self._explain_relevance(query, document)
            recommendation: Recommendation = {'page_content': document.page_content,
                                              'metadata': document.metadata, 'score': score, 'relevance_description': relevance_description}
            recommendations.append(recommendation)
        return recommendations

    def embed_batch(self, texts: list[str], save_dir: str = None) ->  list[list[float]]:
            embeddings = []
            N = len(texts)
            save_batch_id = 1

            for i in tqdm(range(0, N, self.batch_size)):
                st_ind = i
                ed_ind = min(i + self.batch_size, N)
                embeddings_batch = self.embedding_model._embed(texts[st_ind:ed_ind])
                embeddings.extend(embeddings_batch)
                
                if save_dir is not None:
                    # save every 10k samples 
                    if i // 5000 == save_batch_id:
                        utils.save_data_csv(save_dir, embeddings)
                        embeddings = []
                        save_batch_id += 1
                        gc.collect()
                        torch.cuda.empty_cache()
                
            if save_dir is not None:
                # final save
                utils.save_data_csv(save_dir, embeddings)
            else:
                return embeddings


    def embed(self, text: str) -> list[float]:
        ''' embed text into an embedding '''
        embedding = self.embedding_model._embed([text])[0]
        return embedding

    def get_most_similar(self, query: str, K: int = 5) -> list[Document]:
        docs_with_score = self.vector_store.similarity_search_with_score(query, k=K)
        return docs_with_score

    def _explain_relevance(self, text: str, document: Document) -> str:
        # abstarct is passed to _explain_relevance - as a minimal and short represenation of a document. To Be Improved Later
        return ''


if __name__ == "__main__":

    es = EmbeddingStore()
    query = "I am trying to  build a RAG system, what LLM literature should I read?"
    recommendations = es.article_recommendations(query)
    for rec in recommendations:
        print(rec['score'], rec['metadata']['title'])
