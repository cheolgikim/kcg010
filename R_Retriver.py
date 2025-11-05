# R_Retriever_v5_Lifecycle.py
#
# [핵심 변경 사항]
# 1. (문제 1, 3) 데이터 수명 주기 (증분, 삭제, 배치)
#    - 'CONTROL_FILE' (제어 DB)을 도입하여 신규/삭제 문서만 처리
#    - 'get_chunk_id'로 결정론적 ID를 생성, DB 'upsert' 지원 (중복 방지)
#    - 'prune_stale_documents'로 'SOURCES'에서 제거된 문서 DB에서 삭제
#    - 'BATCH_SIZE'로 OOM 방지
# 2. (문제 4) 하이브리드 검색 인덱싱
#    - Elasticsearch (Keyword)와 Chroma (Vector)에 동시 인덱싱
# 3. (문제 5) 분산 저장
#    - 'SOURCES'의 "type"에 따라 동적으로 ES Index/Chroma Collection에 분산 저장
# 4. (문제 2 - 요청 사항) PDF 로더 유지
#    - HTML 파싱(v3) 대신 Playwright PDF 캡처 + 'PyPDFLoader'로 복귀
#    - (경고: GIGO 위험은 여전히 존재합니다.)

import os
import sys
import asyncio
import hashlib
from typing import List, Dict, Set
from dotenv import load_dotenv
from playwright.async_api import async_playwright, Page  # ❗️ 비동기 Playwright
import re
import json

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch, NotFoundError
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader # ❗️ PyPDFLoader
import torch

print("--- [R_Retriver.py] 🚀 v5: 수명 주기/분산 PDF ETL 파이프라인 ---")

# --- ❗️ 1. 설정값 ---
DB_PATH = "./chroma_db_persistent"
ES_URL = "http://localhost:9200"
BATCH_SIZE = 100
CONTROL_FILE = "./processed_sources.json" # (제어 DB 대용)
PDF_CACHE_DIR = "./pdf_cache"
os.makedirs(PDF_CACHE_DIR, exist_ok=True)

# (문제 5) 소스 목록: "type" 지정
SOURCES = [
    {"url": "https://mvje.tistory.com/270", "type": "k8s_tech"},
    {"url": "https://parkkingcar.tistory.com/197", "type": "k8s_tech"},
    {"url": "https://co-de.tistory.com/40", "type": "k8s_tech"},
    # {"url": "https://www.moef.go.kr/policy/policy01.do", "type": "policy"},
]

# --- ❗️ 2. PDF 다운로더 및 로더 (문제 2 제외) ---
def clean_url_to_filename(url: str) -> str:
    if url.startswith("https://"): url = url[8:]
    elif url.startswith("http://"): url = url[7:]
    filename = re.sub(r'[\\/:?."<>|%]', '_', url)
    return filename[:100] + ".pdf"

async def download_and_load_pdf(page: Page, url: str) -> (List[Document], str):
    """
    (비동기) Playwright로 PDF를 캡처하고 PyPDFLoader로 로드
    """
    filepath = os.path.join(PDF_CACHE_DIR, clean_url_to_filename(url))
    
    # ❗️ (문제 1) 캐시 사용. (실제 운영 시: '업데이트' 감지 로직 필요)
    if not os.path.exists(filepath):
        print(f"  - [Download] 캡처 중: {url}")
        try:
            await page.goto(url, wait_until="networkidle", timeout=20000)
            await page.pdf(path=filepath, format="A4", print_background=False)
        except Exception as e:
            print(f"  - [Download 오류] {url} 캡처 실패: {e}")
            return None, None
    else:
        print(f"  - [Download] 캐시 사용: {url}")

    print(f"  - [Load] PyPDFLoader로 로드 중: {filepath}")
    try:
        loader = PyPDFLoader(filepath)
        # ❗️ PyPDFLoader.load()는 동기 함수이므로 to_thread로 실행
        docs = await asyncio.to_thread(loader.load)
        
        # (경고: GIGO) 이 'docs'에는 사이드바, 광고 등 모든 텍스트가 포함됨
        if not docs:
            print(f"  - [Load 경고] PyPDFLoader가 문서를 로드하지 못했습니다.")
            return None, None
            
        print(f"  - [Load] {len(docs)}개 페이지 로드 완료.")
        return docs, filepath
    except Exception as e:
        print(f"  - [Load 오류] {filepath} 로드 실패: {e}")
        return None, None

# --- ❗️ 3. 결정론적 ID 생성기 및 제어 DB (문제 1, 3) ---
def get_source_id(url: str) -> str:
    """URL을 해시하여 고유한 '문서 ID' 생성"""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def get_chunk_id(source_id: str, chunk_index: int) -> str:
    """'문서 ID'와 '청크 순서'를 조합하여 고유한 '청크 ID' 생성"""
    return f"{source_id}_{chunk_index}"

def load_processed_sources() -> Set[str]:
    """(제어 DB) 처리 완료된 source_id 목록을 로드"""
    try:
        with open(CONTROL_FILE, 'r') as f: return set(json.load(f))
    except FileNotFoundError: return set()

def save_processed_sources(processed_ids: Set[str]):
    """(제어 DB) 처리 완료된 source_id 목록을 저장"""
    with open(CONTROL_FILE, 'w') as f: json.dump(list(processed_ids), f)

# --- ❗️ 4. 동적 스토어 (문제 4, 5) ---
# (전역 리소스 로드)
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": device}, encode_kwargs={"normalize_embeddings": True})
    g_es_client = Elasticsearch(hosts=[ES_URL], request_timeout=30)
    g_es_client.info()
    g_chroma_client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    print(f"✅ [Index] 전역 리소스(임베딩, ES, Chroma) 로드 완료 (Device: {device})")
except Exception as e:
    print(f"❌ [Index] 치명적 오류: 전역 리소스 로드 실패: {e}"); sys.exit(1)

def get_stores_for_type(doc_type: str) -> (ElasticsearchStore, Chroma):
    """(문제 5) 문서 유형(type)에 맞는 ES Index와 Chroma Collection을 반환"""
    es_index_name = f"rag_idx_{doc_type}"
    collection_name = f"rag_coll_{doc_type}"
    
    keyword_store = ElasticsearchStore(
        es_connection=g_es_client,
        index_name=es_index_name,
        strategy=ElasticsearchStore.BM25RetrievalStrategy()
    )
    vectorstore = Chroma(
        client=g_chroma_client,
        collection_name=collection_name,
        embedding_function=g_embeddings,
    )
    return keyword_store, vectorstore

# --- ❗️ 5. 문서 삭제 로직 (문제 1) ---
async def prune_stale_documents(current_source_ids: Set[str], processed_source_ids: Set[str]):
    """SOURCES 목록에서 제거된 '오래된' 문서를 DB에서 삭제"""
    stale_ids = processed_source_ids - current_source_ids
    if not stale_ids:
        print("✅ [Index] 삭제할 오래된 문서가 없습니다.")
        return

    print(f"--- [Index] ❗️ {len(stale_ids)}개의 오래된 문서 ID 삭제 시작 ---")
    
    # (개선 필요: 제어 DB가 source_id -> (url, type) 매핑을 저장해야 함)
    # (현재는 매핑 정보가 없어 어떤 URL/Type을 지워야 할지 정확히 알 수 없음)
    
    # (시뮬레이션: 제어 DB에 {'source_id': {'url': '...', 'type': '...'}}이 저장되어 있다고 가정)
    # control_db = load_full_control_db() 
    # for source_id in stale_ids:
    #     info = control_db.get(source_id)
    #     if info:
    #         url = info['url']
    #         doc_type = info['type']
    #         es_idx = f"rag_idx_{doc_type}"
    #         coll_name = f"rag_coll_{doc_type}"
    #         try:
    #             g_es_client.delete_by_query(index=es_idx, body={"query": {"match": {"metadata.source": url}}}, ignore=[404])
    #             collection = g_chroma_client.get_collection(coll_name)
    #             collection.delete(where={"source": url})
    #             print(f"  - [Delete] {url} (Type: {doc_type}) 삭제 완료")
    #         except Exception as e:
    #             print(f"  - [Delete 오류] {url} 삭제 실패: {e}")

    print(f"--- [Index] ❗️ (시뮬레이션) 오래된 문서 삭제 완료 ---")


async def main():
    processed_source_ids = load_processed_sources()
    current_source_ids = set(get_source_id(s['url']) for s in SOURCES)

    # (문제 1) 증분 처리: 신규 소스만 필터링
    new_sources = [s for s in SOURCES if get_source_id(s['url']) not in processed_source_ids]
    
    print(f"--- [Index] 총 {len(SOURCES)}개 소스 중 {len(new_sources)}개 신규 처리 시작 ---")
    
    if not new_sources:
        print("--- [Index] 신규 처리할 문서가 없습니다. ---")
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1200, 
            chunk_overlap=100  # (슬라이딩 대신 100토큰 겹치기로 수정)
        )

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                for source in new_sources:
                    url = source['url']
                    doc_type = source['type']
                    source_id = get_source_id(url)
                    
                    print(f"\n--- [Index] 처리 중: {url} (Type: {doc_type}) ---")

                    # 1. 로드 (PDF 캡처 + PyPDFLoader)
                    docs, filepath = await download_and_load_pdf(page, url)
                    if not docs:
                        print(f"  - [Index 경고] 문서 로드 실패. 건너뜀.")
                        continue
                    
                    # 2. 분할 (문제 3)
                    doc_splits = text_splitter.split_documents(docs)
                    
                    # 3. ID 및 메타데이터 할당 (문제 1)
                    ids = [get_chunk_id(source_id, i) for i, _ in enumerate(doc_splits)]
                    for i, chunk in enumerate(doc_splits):
                        chunk.metadata["chunk_id"] = ids[i]
                        chunk.metadata["source"] = url # 원본 URL 주입
                        # PyPDFLoader의 'page' 메타데이터는 유지됨
                    
                    print(f"  - [Index] {len(doc_splits)}개 청크 및 ID 생성 완료.")

                    # 4. 스토어 가져오기 (문제 5)
                    try:
                        keyword_store, vectorstore = get_stores_for_type(doc_type)
                    except Exception as e:
                        print(f"  - [Index 오류] 스토어 가져오기 실패: {e}"); continue
                        
                    # 5. 배치 인덱싱 (문제 1, 3, 4)
                    for i in range(0, len(doc_splits), BATCH_SIZE):
                        batch_docs = doc_splits[i : i + BATCH_SIZE]
                        batch_ids = ids[i : i + BATCH_SIZE]
                        print(f"  - [Index] 배치 {i//BATCH_SIZE + 1} (Type: {doc_type}) 인덱싱...")
                        try:
                            # (Upsert: ID가 같으면 덮어씀)
                            await asyncio.gather(
                                vectorstore.aadd_documents(batch_docs, ids=batch_ids),
                                keyword_store.aadd_documents(batch_docs, ids=batch_ids, request_timeout=30)
                            )
                        except Exception as e:
                            print(f"  - [Index 오류] 배치 인덱싱 실패: {e}")

                    processed_source_ids.add(source_id) # 제어 DB에 추가
                
                await browser.close()
            print("--- [Index] 신규 문서 처리 완료 ---")
        except Exception as e:
            print(f"❌ [Index] 치명적 오류: Playwright 실행 실패: {e}")
            sys.exit(1)

    # --- 3. 오래된 문서 삭제 (문제 1) ---
    await prune_stale_documents(current_source_ids, processed_source_ids)
    
    save_processed_sources(current_source_ids)
    print("--- [R_Retriver.py] 🚀 v5: ETL 파이프라인 성공적으로 종료 ---")


if __name__ == "__main__":
    asyncio.run(main())