import constants
import time

from elasticsearch import Elasticsearch
import tensorflow_hub as hub

# import pdb

INDEX_NAME = "officequotes"
BATCH_SIZE = constants.BATCH_SIZE
SEARCH_SIZE = constants.SEARCH_SIZE
GPU_LIMIT = constants.GPU_LIMIT

def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return

def profile(fn, *args):
    search_start = time.time()
    response = fn(*args)
    search_time = time.time() - search_start

    return (response, search_time)    

def dummy_search(search_query):
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

    return response

def brute_force_knn(search_query):
    embedding_start = time.time()
    query_vector = embed_text([search_query])[0]
    embedding_time = time.time() - embedding_start

    # print("embedding time: {:.2f} ms".format(embedding_time * 1000))

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    response = client.search(
        index=INDEX_NAME,
        size= SEARCH_SIZE,
        query=script_query,
        _source= {"includes": ["title", "body"]}
    )

    return response    

def knn_search(search_query):
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

    return response

def handle_query():
    query = input("Enter query: ")

    (naive_response, naive_time) = profile(dummy_search, query)
    # (brute_response, brute_time) = profile(brute_force_knn, query)
    (semantic_response, semantic_time) = profile(knn_search, query)

    print('===================================>')
    print(f'Naive time: {naive_time}, approximate KNN {semantic_time}')
    print()

    # print out some results
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
    print("Downloading pre-trained embeddings from tensorflow hub...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Done.")

    client = Elasticsearch(["https://localhost:9200"], http_auth=("elastic", "banana"), verify_certs=False, ssl_show_warn=False)
    client.indices.refresh(index=INDEX_NAME)

    run_query_loop()

    print("Done.")
