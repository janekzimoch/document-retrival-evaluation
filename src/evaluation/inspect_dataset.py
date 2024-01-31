import json

# Path to the JSON file
json_file_path = '/home/zimochpamela/data/mlreference/DORIS-MAE_dataset_v1.json'


def print_json_structure(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        # INSPECT DATA STRUCTURE
        print('keys: ', data.keys())
        print()
        print('QUERY - 100 queries')
        print(f'list length: {len(data["Query"])}')
        print('keys: ', data["Query"][0].keys())
        print('Example:')
        print('query_text: ', data["Query"][0]['query_text'])
        print('article candidate_pool: ', data["Query"][0]['candidate_pool'])
        print()
        print('CORPUS - paper abstracts')
        print('keys: ', data['Corpus'][0].keys())
        print('Example: ', data['Corpus'][0])
        print()
        print('ANNOTATIONS')
        print('keys: ', data['Annotation'][100].keys())
        print('Example: ', data['Annotation'][100])

        # INSPECT TOP 5 ABSTRACTS AND BOTTOM 5 ABSTRACTS FOR SAMPLE QUERY
        # i = 0
        # query = data['Query'][0]['query_text']
        # top_5_abstracts_ids = data['Query'][0]['candidate_pool'][:5]
        # bottom_5_abstracts_ids = data['Query'][0]['candidate_pool'][-5:]
        # top_5_abstracts = [data['Corpus'][id]['original_abstract'] for id in top_5_abstracts_ids]
        # bottom_5_abstracts = [data['Corpus'][id]['original_abstract'] for id in bottom_5_abstracts_ids]

        # print('Query: ', query, '\n')
        # print('TOP 5 ABSTRACTS')
        # for i, abstract in enumerate(top_5_abstracts):
        #     print(f'{i}. - {abstract} \n')
        # print('\n\n BOTTOM 5 ABSTRACTS')
        # for i, abstract in enumerate(bottom_5_abstracts):
        #     print(f'{i}. - {abstract} \n')

if __name__ == '__main__':
    print_json_structure(json_file_path)
