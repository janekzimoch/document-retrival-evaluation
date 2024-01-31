from tqdm import tqdm

from interfaces import Document


def convert_DORIS_MAE_to_my_format(data_in):
    '''
    Formats:
   
    - DORIS_MAE data['Corpus'] : list[dicts]  -> [{'masked_abstract': ..., 'original_abstract': ..., 'title': ..., 'url': ..., 'primary_category': ..., 'categories': ..., 'incoming_citations': ..., 'ss_id': ..., 'outgoing_citations': ..., 'abstract_id': ...]}, {...}, ... ]
    
    - My Format: list[Document] -> [{    
        title: str
        authors: list[str]
        abstract: str
        text: list[str]
        pdf: str
        year: int
        conference: str
        keywords: list[str]
        }, {...}, ...]
    '''
    data_out = []
    for doc_in in tqdm(data_in['Corpus']):
        doc_out: list[Document] = {
            'title': doc_in['title'],
            'authors': None,
            'abstract': doc_in['original_abstract'],
            'text': [doc_in['original_abstract']],
            'pdf': doc_in['url'],
            'year': None,
            'conference': None,
            'keywords': None
        }
        data_out.append(doc_out)
    return data_out

