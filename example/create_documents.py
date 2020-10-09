"""
Example script to create elasticsearch documents.
"""

import pandas as pd
from bert_serving.client import BertClient
import argparse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

bc = BertClient(output_fmt='list')


def create_document(doc, emb, index_name='jobsearch'):
    print(index_name)
    return {
        '_op_type': 'index',
        '_index': index_name,
        'text': doc['text'],
        'title': doc['title'],
        'text_vector': emb
    }

#
# def load_dataset(path):
#     docs = []
#     df = pd.read_csv(path)
#     for row in df.iterrows():
#         series = row[1]
#         doc = {
#             'title': series.Title,
#             'text': series.Description
#         }
#         docs.append(doc)
#     return docs


def bulk_predict(docs, batch_size=256):
    """Predict bert embeddings."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        embeddings = bc.encode([doc['text'] for doc in batch_docs])
        for emb in embeddings:
            yield emb


def main(args):
    client = Elasticsearch()
    # docs = load_dataset(args.data)
    docs=[{'title': 'Bulbasaur', 'text': 'Bulbasaur can be seen napping in bright sunlight. '
                                         'There is a seed on its back. By soaking up the suns rays,'
                                         ' the seed grows progressively larger.'},
  {'title': 'Ivysaur', 'text': 'There is a bud on this Pokémons back. To support its weight, '
                               'Ivysaurs legs and trunk grow thick and strong. If it starts spending more time lying in the sunlight, '
                               'its a sign that the bud will bloom into a large flower soon.'},
  {'title': 'Venusaur', 'text': 'There is a large flower on VENUSAURs back. '
                                'The flower is said to take on vivid colors if it gets plenty of nutrition and sunlight. The flowers aroma soothes the emotions of people.'},
  {'title': 'Charmander', 'text': 'The flame that burns at the tip of its tail is an indication of its emotions.'
                                  ' The flame wavers when CHARMANDER is enjoying itself. If the POKéMON becomes enraged, the flame burns fiercely.'},
  {'title': 'Charmeleon', 'text': 'Charmeleon mercilessly destroys its foes using its sharp claws.'
                                  ' If it encounters a strong foe, it turns aggressive. '
                                  'In this excited state, the flame at the tip of its tail flares with a bluish white color.'}]

    print("This is the generated documents",docs)
    to_bulk=[]
    for doc, emb in zip(docs, bulk_predict(docs)):
        print("in the doc loop")
        d = create_document(doc, emb, args.index_name)
        to_bulk.append(d)
        # Elasticsearch.index(index=args.index_name,body=d)
    print(to_bulk)
    bulk(client, to_bulk)

            # f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    # parser.add_argument('--data', help='data for creating documents.')
    # parser.add_argument('--save', default='documents.jsonl', help='created documents.')
    parser.add_argument('--index_name', default='jobsearch', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)

# python create_documents.py --index_name=jobsearch
