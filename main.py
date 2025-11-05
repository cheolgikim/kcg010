# RAG.py (수정됨: Public 다국어 Reranker 및 경고 수정)

import antigravity
import getpass
from math import e
import operator
from threading import currentThread
from typing import Annotated, TypedDict, List, Tuple, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import gradio as gr
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import asyncio
import hashlib #중복제거 로직 위한 hashlib
from langgraph.graph import StateGraph, END

from langchain_core.documents import Document
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
import logging
import sys
import langchain

# --- ❗️ 1. 아키텍처 분리를 위한 추가 Import ---
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
# ❗️ (langchain-chroma 설치 확인)
from langchain_chroma import Chroma  
from langchain_community.retrievers import BM25Retriever
import torch
from elasticsearch import Elasticsearch
from langchain_community.vectorstores import ElasticsearchStore # ES Store 사용을 위해 추가
from langchain_elasticsearch import BM25RetrievalStrategy
# ❗️ EnsembleRetriever 임포트 제거됨

# --- ❗️ 2. Reranker(Grader 대체)를 위한 추가 Import ---
from sentence_transformers import CrossEncoder
import numpy as np
# -----------------------------------------------

# --- 로깅 설정 (원본 유지) ---
langchain.debug = False
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers = [] 
file_handler = logging.FileHandler('RAG.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
langchain_logger = logging.getLogger('langchain')
langchain_logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("로깅 시스템 설정 완료.")
# -----------------------------

load_dotenv()

# --- ❗️ 3. 전역 설정 변수 및 리소스 초기화 (NameError 해결) ---
DB_PATH = "./chroma_db_persistent"
ES_URL = "http://localhost:9200"
# ❗️ R_Retriever.py의 소스와 일치하도록 설정 (현재 'k8s_tech'만 인덱싱 가능)
ALL_DOC_TYPES = ["k8s_tech"] 
CHROMA_COLLECTION_PREFIX = "rag_coll_" 
ES_INDEX_PREFIX = "rag_idx_" 
RRF_K = 60 

g_embeddings = None
g_chroma_client = None
g_es_client = None
g_reranker = None

# --- 3-1. 전역 리소스 로드 ---
try:
    # 임베딩 모델 로드 (전역 변수 g_embeddings)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device}, 
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info(f"✅ [RAG.py] 임베딩 모델 로드 완료 (Device: {device})")

    # 전역 ChromaDB 및 Elasticsearch 클라이언트 로드
    g_es_client = Elasticsearch(hosts=[ES_URL], request_timeout=30)
    g_es_client.info()
    
    g_chroma_client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    logger.info("✅ [RAG.py] ES 및 Chroma 클라이언트 로드 완료.")
    
except Exception as e:
    logger.critical(f"❌ [RAG.py] 치명적 오류: 전역 리소스 로드 실패. ES 또는 ChromaDB 실행 여부를 확인하세요. {e}", exc_info=True)
    sys.exit(1)

# --- ❗️ 3. 전역 리트리버 및 Reranker 설정 ---

def setup_retrievers() -> Tuple[Dict[str, Chroma], Dict[str, BM25Retriever]]:
    """
    모든 doc_type에 대한 분산 Chroma 및 BM25 리트리버를 초기화합니다.
    """
    
    chroma_retrievers = {}
    for doc_type in ALL_DOC_TYPES:
        collection_name = f"{CHROMA_COLLECTION_PREFIX}{doc_type}"
        try:
            vectorstore = Chroma(
                client=g_chroma_client,
                collection_name=collection_name,
                embedding_function=g_embeddings,
            )
            chroma_retrievers[doc_type] = vectorstore.as_retriever(search_kwargs={"k": 10}) 
            logger.info(f"✅ Chroma Retriever 초기화: {collection_name}")
        except Exception as e:
            logger.warning(f"⚠️ Chroma 컬렉션 {collection_name} 초기화 실패 (Skip): {e}")

    bm25_retrievers = {}
    for doc_type in ALL_DOC_TYPES:
        # ❗️ R_Retriever.py의 인덱싱 규칙에 따라 _keyword 접미사를 붙입니다.
        index_name = f"{ES_INDEX_PREFIX}{doc_type}_keyword" 
        try:
            keyword_store = ElasticsearchStore(
                es_connection=g_es_client,
                index_name=index_name,
                strategy=BM25RetrievalStrategy()
            )
            # 2단계: ElasticsearchStore 인스턴스에서 BM25 리트리버를 명시적으로 얻어냅니다.
            # ❗️ 여기서 get_retriever 메서드가 BM25를 반환한다고 가정합니다.
            bm25_retrievers[doc_type] = keyword_store.as_retriever(search_kwargs={"k": 10})
            logger.info(f"✅ BM25 Retriever 초기화: {index_name}")

        except Exception as e:
            logger.warning(f"⚠️ ES 인덱스 {index_name} 초기화 실패 (Skip): {e}")


    if not chroma_retrievers and not bm25_retrievers:
        logger.error("모든 리트리버 초기화에 실패했습니다. 시스템 종료.")
        sys.exit(1)

    return chroma_retrievers, bm25_retrievers

def setup_reranker():
    """Public 다국어 Reranker 모델을 로드합니다."""
    try:
        logger.info("--- [RAG.py] CrossEncoder (Reranker) 모델 로드 시작 ---")
        model_name = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
        reranker = CrossEncoder(model_name, max_length=512)
        logger.info(f"✅ [RAG.py] CrossEncoder (Reranker) 모델 로드 완료 ({model_name}).")
        return reranker
    except Exception as e:
        logger.error(f"❌ [RAG.py] CrossEncoder 모델 로드 실패: {e}", exc_info=True)
        logger.warning("Reranker 로드 실패. Reranking 없이 진행됩니다.")
        return None

# ❗️ 애플리케이션 시작 시 전역 변수 초기화
try:
    g_chroma_retrievers, g_bm25_retrievers = setup_retrievers() # ❗️ 딕셔너리로 저장
    g_reranker = setup_reranker()
except Exception as e:
    logger.critical(f"❌ [RAG.py] 리트리버/Reranker 초기화 실패. 애플리케이션을 시작할 수 없습니다. {e}", exc_info=True)
    sys.exit(1)
# -----------------------------------------------


# --- ❗️ 4. [신규] 수동 RRF(Reciprocal Rank Fusion) 로직 ---
async def manual_ensemble_retriever(question: str, k: int = 20) -> List[Document]:
    """
    모든 분산된 Chroma 및 BM25 리트리버에서 병렬로 검색하고 RRF를 적용합니다.
    """
    logger.debug(f"수동 Ensemble Retriever 실행 (분산): '{question}'")
    
    # 1. 모든 검색기에서 비동기 병렬 검색 (Vector + Keyword)
    search_tasks = []
    
    # Chroma 검색 작업 추가 (모든 doc_type)
    for doc_type, retriever in g_chroma_retrievers.items():
        task = retriever.aget_relevant_documents(question)
        search_tasks.append(task)
        logger.debug(f"[Search Task] Chroma-{doc_type} 추가됨")
        
    # BM25 검색 작업 추가 (모든 doc_type)
    for doc_type, retriever in g_bm25_retrievers.items():
        task = retriever.aget_relevant_documents(question)
        search_tasks.append(task)
        logger.debug(f"[Search Task] BM25-{doc_type} 추가됨")

    # 모든 검색을 병렬로 실행
    results = await asyncio.gather(*search_tasks, return_exceptions=True)
    
    # 2. Document 고유 ID 생성 (RRF를 위한 안정적인 키)
    def get_doc_key(doc: Document) -> str:
        """문서 내용을 기반으로 고유 키를 생성합니다."""
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        content_hash = hashlib.sha256(doc.page_content[:500].encode('utf-8')).hexdigest()
        return f"{source}_{page}_{content_hash}"
    
    # 3. RRF (Reciprocal Rank Fusion) 적용
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    # 결과를 리트리버별로 나누어 랭킹을 계산합니다.
    for i, retriever_results in enumerate(results):
        if isinstance(retriever_results, Exception):
            logger.error(f"병렬 검색 작업 {i} 실패: {retriever_results}")
            continue

        # 랭크를 1부터 시작
        for rank, doc in enumerate(retriever_results, 1):
            doc_key = get_doc_key(doc)
            doc_map[doc_key] = doc
            
            # RRF 점수 계산: 1 / (k + rank)
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (RRF_K + rank)

    # 4. 점수를 기준으로 내림차순 정렬 및 상위 k개 선택
    sorted_scores = sorted(rrf_scores.items(), key=itemgetter(1), reverse=True)
    
    final_docs = []
    for doc_key, score in sorted_scores[:k]:
        doc = doc_map[doc_key]
        doc.metadata["rrf_score"] = round(score, 4) # 디버깅을 위해 점수 추가
        final_docs.append(doc)

    logger.info(f"✅ RRF 통합 검색 완료. 최종 {len(final_docs)}개 문서 반환.")
    return final_docs
# ---------------------------------------------


# --- 5. RAG 프롬프트 및 LLM 설정 

# ❗️ [수정] stream=True 파라미터를 제거하여 UserWarning 제거
# ❗️ .astream() 호출 시 자동으로 스트리밍이 활성화됩니다.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an AI assistant helping with Question-Answering (QA) tasks within a Retrieval-Augmented Generation (RAG) system.

Your mission is to answer the user's **LATEST QUESTION** based on the provided **CONTEXT** and the ongoing **CHAT HISTORY**.
Please formulate your answer to be as **helpful and clear as possible**, synthesizing relevant information found in the context. Use the chat history to understand the conversation flow.

### Answer Generation Guidelines:

1.  **Context-Based Answers:** Your answer **must be based solely** on the information present in the **PROVIDED CONTEXT**. Do not use any external knowledge.
2.  **Synthesize Information:** Even if the context doesn't contain an exact match for the question, you can **synthesize** or **explain** using related information found within the context.
3.  **Cite Sources:** Clearly state the **source(s)** from the context that support your answer at the end.
    * Format: `**Source:**\n- [File Name(Page Number)] or [URL]`
    * Omit the source section if no supporting information was found in the context.
4.  **Handle Unanswerable Questions:** If you cannot find the answer to the question within the CONTEXT and CHAT HISTORY, honestly state: "Based on the provided documents, I could not find the information to answer that question."
5.  **Language:** You **MUST** answer in **Korean**.

###

# CHAT HISTORY:
{chat_history}

# CONTEXT to use for answering the LATEST question:
{context}

# LATEST User QUESTION:
{question}

# Final ANSWER (in Korean) based on CHAT HISTORY and CONTEXT:
"""),
    ]
)
# (질문 재작성, LLM 체인, ChatState 원본 유지)
async def format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    if not chat_history:
        return "No chat history."
    context_parts = []
    for user_msg, bot_msg in chat_history:
        context_parts.append(f"User: {user_msg}\nAssistant: {bot_msg}")
    return "\n".join(context_parts)

condense_question_prompt=ChatPromptTemplate.from_messages(
    [
        ("system","Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question."),
        ("human", "Chat History:\n{chat_history}\n\nFollow Up Input: {question}\n\nStandalone question:"),
    ]
)
condense_question_chain = condense_question_prompt | llm | StrOutputParser()
generator_chain = ( rag_prompt | llm | StrOutputParser() )
class ChatState(TypedDict):
    question: str
    chat_history: Annotated[List[Tuple[str, str]], operator.add]
    standalone_question: str
    context: Annotated[List[Document], operator.add]
    ai_response: Annotated[str, operator.add]

# --- ❗️ 6. Reranker 로직 (비동기 래핑) ---
async def run_reranker(question: str, documents: List[Document], top_n: int = 3):
    # (원본 유지, 전역 'g_reranker' 사용)
    if not g_reranker: 
        logger.warning("Reranker가 로드되지 않았습니다. 상위 N개 문서를 그대로 반환합니다.")
        return documents[:top_n]
        
    logger.debug(f"Reranking {len(documents)}개 문서 (Top {top_n} 선택)...")
    
    pairs = [(question, doc.page_content) for doc in documents]
    
    try:
        scores = await asyncio.to_thread(g_reranker.predict, pairs) 
        logger.debug(f"Reranker 점수 계산 완료 (개수: {len(scores)})")
    except Exception as e:
        logger.error(f"Reranker predict 실패: {e}", exc_info=True)
        return documents[:top_n]

    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True) 
    
    reranked_docs = [doc for doc, score in doc_scores[:top_n]]
    
    logger.info("=============Reranker 기반 문서 필터링===============")
    for i, (doc, score) in enumerate(doc_scores):
        if i < top_n:
            logger.info(f"결과: [Top {i+1}, Score: {score:.4f}] - {doc.metadata.get('source', 'N/A')}")
        else:
            logger.debug(f"결과: [탈락, Score: {score:.4f}] - {doc.metadata.get('source', 'N/A')}")
    logger.info("=================================================")

    return reranked_docs
# ---------------------------------------------

# --- ❗️ 7. RAG 파이프라인 노드 수정 ---
async def rephrase_question_node(state: ChatState):
    # (원본 유지)
    logger.info("--- 1. [NODE] rephrase_question_node ---")
    logger.debug(f"Entering state: {state}")
    chat_history_str =await format_chat_history(state["chat_history"])
    standalone_question = ""
    async for chunk in condense_question_chain.astream({
        "chat_history": chat_history_str,
        "question": state["question"]
    }):
        standalone_question += chunk
    logger.info(f"재작성된 질문: {standalone_question}")
    return {"standalone_question": standalone_question}

async def retrieve_documents_node(state: ChatState):
    """2. ❗️ [수정] 수동 Ensemble 및 Reranking 노드"""
    logger.info("--- 2. [NODE] retrieve_documents_node ---")
    logger.debug(f"Entering state: {state}")
    
    question = state["standalone_question"]
    
    # 1. ❗️ [수정] 수동 Ensemble Retriever 호출 (1단계)
    documents = await manual_ensemble_retriever(question, k=20)
    logger.info(f"1단계 (수동 Ensemble) 검색된 문서 수: {len(documents)}")

    if not documents:
        logger.warning("1단계 검색 결과 문서가 없습니다.")
        return {"context": []}

    # 2. ❗️ Reranker 실행 (2단계)
    TOP_N_RERANK = 3 
    filtered_docs = await run_reranker(question, documents, top_n=TOP_N_RERANK)
    
    if not filtered_docs:
        logger.warning("Reranker 필터링 후 관련 있는 문서를 찾지 못했습니다.")

    return {"context": filtered_docs}

# (format_docs, generate_answer_node, fallback_answer_node, should_generate_answer 원본 유지)
def format_docs(docs: List[Document]) -> str:
    logger.debug(f"Formatting {len(docs)} documents...")
    formatted_strings = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page")
        content = doc.page_content
        doc_string = f"<document><content>{content}</content><source>{source}</source>"
        if page is not None:
            try:
                doc_string += f"<page>{int(page) + 1}</page>"
            except ValueError:
                doc_string += f"<page>{page}</page>"
        doc_string += "</document>"
        formatted_strings.append(doc_string)
    formatted_result = "\n\n".join(formatted_strings)
    logger.debug(f"Formatted Docs (first 1000 chars): {formatted_result[:1000]}...")
    return formatted_result
async def generate_answer_node(state: ChatState):
    logger.info("--- 3. [NODE] generate_answer_node (Streaming) ---")
    logger.debug(f"Entering state: {state}")
    context = state["context"]
    standalone_question = state["standalone_question"]
    chat_history_tuples = state["chat_history"]
    logger.debug(f"Context (doc count: {len(context)}): {context}")
    formatted_context = format_docs(context)
    logger.debug(f"Formatted Context (first 1000 chars): {formatted_context[:1000]}...")
    logger.debug(f"Chat History (tuple count: {len(chat_history_tuples)}): {chat_history_tuples}")
    formatted_chat_history = await format_chat_history(chat_history_tuples)
    logger.debug(f"Formatted Context (first 1000 chars): {formatted_context[:1000]}...")
    logger.debug(f"Formatted Chat History: {formatted_chat_history}")
    try:
        logger.info("Streaming LLM response...")
        async for chunk in generator_chain.astream({
            "question": standalone_question,
            "context": formatted_context,
            "chat_history" : formatted_chat_history
        }):
            logger.debug(f"LLM Chunk (Yielding): {chunk}")
            yield {"ai_response": chunk} 
    except Exception as e:
        logger.error(f"LLM 답변 생성 중 에러 발생: {e}", exc_info=True)
        yield {"ai_response": "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다."}
async def fallback_answer_node(state: ChatState):
    logger.info("--- 4. [NODE] fallback_answer_node ---")
    logger.warning("관련 문서를 찾지 못해 Fallback 노드로 진입합니다.")
    answer = "죄송합니다, 질문과 관련된 문서를 찾지 못했습니다. 더 구체적인 질문으로 시도해 주세요."
    yield {"ai_response": answer}
def should_generate_answer(state: ChatState) -> str:
    logger.info("--- [DECISION] should_generate_answer ---")
    logger.debug(f"Checking context: {state['context']}")
    if state["context"]:
        logger.info("결과: [관련 문서 있음] -> 답변 생성으로 이동")
        return "generate_answer"
    else:
        logger.info("결과: [관련 문서 없음] -> Fallback으로 이동")
        return "fallback_answer"

# --- 8. RAG 그래프 구성 (원본 유지) ---
graphagent = StateGraph(ChatState)
graphagent.add_node("rephrase_question", rephrase_question_node)
graphagent.add_node("retrieve_documents", retrieve_documents_node)
graphagent.add_node("generate_answer", generate_answer_node)
graphagent.add_node("fallback_answer", fallback_answer_node)
graphagent.set_entry_point("rephrase_question")
graphagent.add_edge("rephrase_question", "retrieve_documents")
graphagent.add_conditional_edges(
    "retrieve_documents",
    should_generate_answer, 
    {
        "generate_answer": "generate_answer", 
        "fallback_answer": "fallback_answer", 
    },
)
graphagent.add_edge("generate_answer", END)
graphagent.add_edge("fallback_answer", END)
compiled_graph = graphagent.compile()


# --- 9. Gradio 'chat' 함수 (원본 유지) ---
async def chat(message: str, history: List[List[str]]):
    logger.info("="*50)
    logger.info(f"Gradio 'chat' 함수 시작: Message='{message}'")
    logger.debug(f"History (Gradio format): {history}")
    
    rag_history_tuples: List[Tuple[str, str]] = [tuple(turn) for turn in history]
    logger.debug(f"History (RAG format): {rag_history_tuples}")
    
    current_question = message
    graph_input = {
        "question": current_question, 
        "chat_history": rag_history_tuples,
    }
    
    accumulated_response = ""
    streamed_something = False 
    
    try: 
        async for event in compiled_graph.astream_events(graph_input, version="v1"):
            kind = event["event"]
            name = event.get("name", "")
            logger.debug(f"[Graph Stream Event]: kind={kind}, name={name}")
                
            if kind == "on_chain_stream" and (name == "generate_answer" or name == "fallback_answer"):
                chunk_data = event["data"]["chunk"] 
                if isinstance(chunk_data, dict) and "ai_response" in chunk_data:
                    ai_chunk = chunk_data.get("ai_response")
                    if ai_chunk:
                        accumulated_response += ai_chunk
                        logger.debug(f"Yielding to Gradio (Accumulated): {accumulated_response[:100]}...")
                        yield accumulated_response
                        streamed_something = True
                        
        if not streamed_something:
             logger.warning("스트림에서 아무런 응답도 생성되지 않았습니다.")
             yield "죄송합니다, 응답을 생성하는 데 실패했습니다."
    
    except Exception as e:
        logger.error(f"Gradio 'chat' 함수에서 심각한 에러 발생: {e}", exc_info=True)
        error_message = f"죄송합니다, 그래프 실행 중 심각한 오류가 발생했습니다: {e}"
        if not streamed_something:
             yield error_message
        else:
             yield accumulated_response + f"\n\n[오류 발생]: {e}"

    logger.info(f"Gradio 'chat' 함수 종료 (Full Response Logged: {accumulated_response[:200]}...)")
    logger.info("="*50)


# --- 10. Gradio UI 설정 (❗️ type='messages' 경고 수정) ---
css = "footer {display: none !important;}"
demo = gr.ChatInterface(
    fn=chat,
    title="RAG 시스템 테스트)",
    description="RAG시스템 테스트입니다.",
    theme="soft",
    examples=[["쿠버네티스(k8s)의 네트워크 네임스페이스 격리는 무엇인가요?"], ["k8s의 CNI는 어떤 역할을 하나요?"]],
    submit_btn="질문하기",
    css=css
    # ❗️ 참고: Gradio의 'tuples' 경고를 완전히 없애려면 'chat' 함수의 
    # ❗️ 입/출력 형식을 (message, history)에서 'messages' 딕셔너리 리스트로
    # ❗️ 변경해야 합니다. (현재 작동에는 영향 없음)
)

if __name__ == "__main__":
    # ❗️ 'main.py'로 파일 이름이 변경되었으므로 __main__ 로거 이름을 확인합니다.
    # (현재 __name__ == "__main__" 이므로 로깅은 정상 작동합니다)
    logger.info("Gradio 애플리케이션을 시작합니다...")
    demo.analytics_enabled = False
    demo.launch(show_api=False)