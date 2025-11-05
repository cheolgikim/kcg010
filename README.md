# 1. 설치
sudo apt install mariadb-server
.env 에 OPENAI_API_KEY 입력
(세션 2개 열고 가상환경 2개 생성. R_Retriver.py와 main.py 각각 다른 환경에서 실행하기 위함)
python3.11 -m venv .venv1
source .venv1/bin/activate
python3.11 -m venv .venv2
source .venv2/bin/activate

(가상환경 .venv1 에서 실행 ) pip install -r requirements_RAG.txt
(가상환경 .venv2 에서 실행 ) pip install -r requirements_R_Retriver.txt


# 2. MYSQL서비스 시작
sudo systemctl start mariadb
sudo systemctl enable mariadb

# 3. MYSQL보안 설정 (비밀번호 설정 등)
sudo mysql_secure_installation

# 4. DB 및 사용자 생성 (v9용)
sudo mysql -u root -p

-- MySQL 프롬프트에서 실행
CREATE DATABASE IF NOT EXISTS rag_control_db DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'rag_user'@'localhost' IDENTIFIED BY 'cjfrl91!!';
GRANT ALL PRIVILEGES ON rag_control_db.* TO 'rag_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;

#5. .env 파일에 아래 mysql 정보 추가
#--- MySQL 제어 DB 설정 ---
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=rag_user
MYSQL_PASSWORD=cjfrl91!!
MYSQL_DB=rag_control_db

#6. 엘라스틱서치 도커 띄우기.

docker run -d --name es01 -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "ES_JAVA_OPTS=-Xms1000m -Xmx1000m" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.10.4

#7. 각각 requirements.txt 파일들을 실행한다.

#8. Gradio 버그 해결
pip install --upgrade gradio gradio_client fastapi uvicorn

#9. 코드의 주요 장점 및 기능

1. 데이터 수명 주기 관리: CONTROL_FILE (JSON)을 사용해 이미 처리된 문서를 추적, 신규 문서만 처리(증분 업데이트)하고 SOURCES에서 제거된 문서를 삭제(Pruning)하려는 시도는 대규모 운영의 핵심

2. 하이브리드 인덱싱: Vector(Chroma)와 Keyword(ES)에 동시 저장하여 하이브리드 검색의 기반을 마련

3. 동적 분산 저장: SOURCES의 "type"별로 인덱스/컬렉션을 분리하는 구조는 향후 "법령", "보도자료" 등 도메인별 검색 전략을 다르게 가져갈 수 있는 높은 확장성을 제공

4. 비동기 수집: async_playwright를 사용해 I/O 바운드 작업인 웹페이지 로딩을 병렬 처리하여 수집 속도를 높였다.

5. 결정론적 ID: get_chunk_id로 고유 ID를 생성, upsert (덮어쓰기)가 가능하게 하여 데이터 일관성 관리에 용이

6. 최신 RAG 아키텍처: (Rephrase -> Retrieve -> Rerank -> Generate)는 RAG의 정확도를 극대화하는 표준적이고 강력한 파이프라인

7. 완전 비동기 및 스트리밍: asyncio.gather (검색), asyncio.to_thread (Reranker), astream (LLM) 등 시스템 전반에 비동기/스트리밍을 적용하여 **첫 답변까지의 시간(TTFT)**을 최소화하려는 설계

8. 고급 검색 전략:
        - RRF (Reciprocal Rank Fusion): 여러 리트리버(Vector, Keyword)의 결과를 수동으로 조합하는 RRF는 표준 EnsembleRetriever보다 유연하며 성능이 좋음.
        - Reranking: CrossEncoder를 2단계에서 사용하여, 1단계에서 찾은 문서들이 "실제로 질문에 답이 되는지"를 판단하는 로직은 답변 정확도를 비약적으로 향상

9. LangGraph 활용: 검색 결과 유무에 따라 답변/폴백으로 분기하는 로직을 StateGraph로 명확하게 관리

10. OCR은 GooGleOCR이 성능이 가장 좋다고 판단해서 구글 ocr을 사용했다.

