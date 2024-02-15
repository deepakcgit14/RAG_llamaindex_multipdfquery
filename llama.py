## Retrieval augmented generation

import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
documents=SimpleDirectoryReader("data").load_data()


index=VectorStoreIndex.from_documents(documents,show_progress=True)
query_engine=index.as_query_engine()

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

retriever=VectorIndexRetriever(index=index,similarity_top_k=4)
postprocessor=SimilarityPostprocessor(similarity_cutoff=0.80)

query_engine=RetrieverQueryEngine(retriever=retriever,
                                  node_postprocessors=[postprocessor])

response=query_engine.query("What is attention is all you need?")

from llama_index.response.pprint_utils import pprint_response
pprint_response(response,show_source=True)
print(response)