import json
import time

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import tensorflow_hub as hub

# import pdb

##### INDEXING #####

def index_data():
    print("Creating the 'posts' index.")
    client.indices.delete(index=INDEX_NAME, ignore=[404])

    with open(INDEX_FILE) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=INDEX_NAME, mappings=eval(source)['mappings'], settings=eval(source)['settings'])

    docs = []
    count = 0

    with open(DATA_FILE) as data_file:
        for line in data_file:
            line = line.strip()

            doc = json.loads(line)
            if doc["type"] != "question":
                continue

            docs.append(doc)
            count += 1

            if count % BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print("Indexed {} documents.".format(count))

        if docs:
            index_batch(docs)
            print("Indexed {} documents.".format(count))

    client.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")

def index_batch(docs):
    # pdb.set_trace()
    titles = [doc["title"] for doc in docs]
    bodies = [doc["body"] for doc in docs]
    title_vectors = embed_text(titles)
    body_vectors = embed_text(bodies)

    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["title_vector"] = title_vectors[i]
        request["body_vector"] = body_vectors[i]
        requests.append(request)

    # pdb.set_trace()
    bulk(client, requests)

##### SEARCHING #####

def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return

def dummy_search(search_query):
    search_start = time.time()

    # dummy search
    query = {
        "multi_match": {
            "query": search_query,
            "fields": [
                "title",
                "body"
            ]
        }        
    }

    response = client.search(
        index=INDEX_NAME,
        size= SEARCH_SIZE,
        query=query,
        _source= {"includes": ["title", "body"]}
    )    

    search_time = time.time() - search_start
    print(f'naive search time {search_time * 1000} ms')

    return (response, search_time)

def knn_search(search_query):
    search_start = time.time()

    embedding_start = time.time()
    query_vector = embed_text([search_query])[0]
    embedding_time = time.time() - embedding_start

    # print("embedding time: {:.2f} ms".format(embedding_time * 1000))

    script_query = {
        "field": "body_vector",
        "query_vector": query_vector,
        "k": 10,
        "num_candidates": 100            
    }

    response = client.knn_search(
        index=INDEX_NAME,
        knn=script_query,
        _source= {"includes": ["title", "body"]}
    )

    search_time = time.time() - search_start
    print(f'knn search time {search_time * 1000} ms')    

    return (response, search_time)

def handle_query():
    query = input("Enter query: ")

    # brute force KNN
    # script_query = {
    #     "script_score": {
    #         "query": {"match_all": {}},
    #         "script": {
    #             "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
    #             "params": {"query_vector": query_vector}
    #         }
    #     }
    # }

    # response = client.search(
    #     index=INDEX_NAME,
    #     size= SEARCH_SIZE,
    #     query=script_query,
    #     _source= {"includes": ["title", "body"]}
    # )

    # print()
    # print("{} total hits.".format(response["hits"]["total"]["value"]))
    # print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    # print("search time: {:.2f} ms".format(search_time * 1000))

    (naive_response, naive_time) = dummy_search(query)
    (semantic_response, semantic_time) = knn_search(query)
    # print(f'time difference {semantic_time / naive_time}')

    for hit in semantic_response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"]['title'])
        print(hit["_source"]['body'])
        print()

##### EMBEDDING #####

def embed_text(text):
    vectors = embed(text)
    return [vector.numpy().tolist() for vector in vectors]

##### MAIN SCRIPT #####

if __name__ == '__main__':
    INDEX_NAME = "posts2"
    INDEX_FILE = "data/posts/index.json"

    DATA_FILE = "data/posts/posts.json"
    BATCH_SIZE = 1000

    SEARCH_SIZE = 5

    GPU_LIMIT = 0.5

    print("Downloading pre-trained embeddings from tensorflow hub...")
    # embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    embeddings = embed([''])
    print("Done.")

    client = Elasticsearch(["https://localhost:9200"], http_auth=("elastic", "banana"), verify_certs=False, ssl_show_warn=False)
    client.indices.refresh(index=INDEX_NAME)

    # index_data()
    run_query_loop()

    print("Done.")
