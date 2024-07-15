import certifi
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import config


ca = certifi.where()
loaders = [
    WebBaseLoader("https://onetech.vn/cong-ty-onetech-asia"),
    WebBaseLoader("https://onetech.vn/cong-ty-onetech-asia/lich-su-phat-trien"),
    WebBaseLoader("https://onetech.vn/cong-ty-onetech-asia/su-menh-tam-nhin-gia-tri")
]

data = []
for loader in loaders:
    data.extend(loader.load())

print(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
    "\n\n", "\n", "(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(data)

print('Split into ' + str(len(docs)) + ' docs')

embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)

client = MongoClient(config.mongodb_conn_string, tlsCAFile=ca)

collection = client[config.db_name][config.collection_name]

collection.delete_many({})

docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=config.index_name
)
