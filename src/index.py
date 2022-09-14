import constants
import json
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# import pdb

# INDEX_NAME = "posts"
# INDEX_FILE = "data/posts/index.json"
# DATA_FILE = "data/posts/posts.json"

BATCH_SIZE = constants.BATCH_SIZE
SEARCH_SIZE = constants.SEARCH_SIZE
GPU_LIMIT = constants.GPU_LIMIT

def embed_text(text):
    vectors = embed(text)
    return [vector.numpy().tolist() for vector in vectors]

def index_data():
    for index_name in os.listdir('data'):
        if index_name == 'wikibooks' or index_name == '.DS_Store':
            continue

        print(f'Creating the "{index_name}" index')
        index_file_path = f'data/{index_name}/index.json'
        data_file_path = f'data/{index_name}/data.json'

        client.indices.delete(index=index_name, ignore=[404])

        with open(index_file_path) as index_file:
            source = index_file.read().strip()
            client.indices.create(index=index_name, mappings=eval(source)['mappings'], settings=eval(source)['settings'])

        docs = []
        count = 0

        with open(data_file_path) as data_file:
            for line in data_file:
                line = line.strip()

                doc = json.loads(line)
                if doc["type"] != "question":
                    continue

                docs.append(doc)
                count += 1

                if count % BATCH_SIZE == 0:
                    index_batch(docs, index_name)
                    docs = []
                    print("Indexed {} documents.".format(count))

            if docs:
                index_batch(docs, index_name)
                print("Indexed {} documents.".format(count))

        # client.indices.refresh(index=INDEX_NAME)
        print("Done indexing.")

def index_batch(docs, index_name):
    # pdb.set_trace()
    titles = [doc["title"] for doc in docs]
    bodies = [doc["body"] for doc in docs]
    title_vectors = embed_text(titles)
    body_vectors = embed_text(bodies)

    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = index_name
        request["title_vector"] = title_vectors[i]
        request["body_vector"] = body_vectors[i]
        requests.append(request)

    # pdb.set_trace()
    bulk(client, requests)

if __name__ == '__main__':
    print("Downloading pre-trained embeddings from tensorflow hub...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Done.")

    client = Elasticsearch(["https://localhost:9200"], http_auth=("elastic", "banana"), verify_certs=False, ssl_show_warn=False)

    index_data()
    print("Done.")
