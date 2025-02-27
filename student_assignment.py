import datetime
import chromadb
import traceback
import csv
import time
from datetime import datetime

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    print("設置 Collection 名稱和 metadata 參數")
    # 設置 Collection 名稱和 metadata 參數
    COLLECTION_NAME = "TRAVEL"
    METADATA_PARAMS = {"hnsw:space": "cosine"}
    CSV_FILE = "COA_OpenData.csv"

    print("初始化 ChromaDB 客戶端")
    # 初始化 ChromaDB 客戶端
    client = chromadb.PersistentClient(path="./chromadb_data")

    # 創建或獲取 Collection
    print("創建或獲取 Collection")
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata=METADATA_PARAMS)

    def convert_to_timestamp(date_str):
        """將日期字串轉換為時間戳（秒）"""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return int(time.mktime(dt.timetuple()))
        except ValueError:
            return None

    # 讀取 CSV 檔案並寫入 ChromaDB
    print("讀取 CSV 檔案並寫入 ChromaDB")
    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        print(f"總行數: {len(rows)}")
        for row in rows:
            print(row)
            metadata = {
                "file_name": CSV_FILE,
                "name": row.get("Name", ""),
                "type": row.get("Type", ""),
                "address": row.get("Address", ""),
                "tel": row.get("Tel", ""),
                "city": row.get("City", ""),
                "town": row.get("Town", ""),
                "date": convert_to_timestamp(row.get("CreateDate", "")),
            }
            
            document = row.get("HostWords", "")
            
            # 生成 ID
            doc_id = row.get("id", str(time.time_ns()))
            
            collection.add(ids=[doc_id], documents=[document], metadatas=[metadata])
    return collection

    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection


generate_hw01()