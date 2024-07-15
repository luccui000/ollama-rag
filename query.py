import config
import certifi
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAIEmbeddings, OpenAI
import warnings

ca = certifi.where()

warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")

# query = "Thông tin về Công ty Cổ Phần OneTechASIA?"
# query = "Sứ mệnh tầm nhìn của công ty"
query = "Lịch sử phát triển của công ty"
print(query)

# Initialize MongoDB python client
client = MongoClient(config.mongodb_conn_string, tlsCAFile=ca)
collection = client[config.db_name][config.collection_name]

# initialize vector store
vectorStore = MongoDBAtlasVectorSearch(
    collection, OpenAIEmbeddings(openai_api_key=config.openai_api_key), index_name=config.index_name
)

docs = vectorStore.max_marginal_relevance_search(query, K=5)
print(docs)

print(docs[0].metadata['title'])
print(docs[0].page_content)

# Contextual Compression
llm = OpenAI(openai_api_key=config.openai_api_key, temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorStore.as_retriever()
)

compressed_docs = compression_retriever.get_relevant_documents(query)
print(compressed_docs[0].metadata['title'])
print(compressed_docs[0].page_content)
