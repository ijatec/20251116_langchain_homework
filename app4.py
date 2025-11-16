import os
import shutil
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# -------------------------------
# 환경 설정
# -------------------------------
load_dotenv(".env")

VECTORSTORE_DIR = "faiss_index"
MAX_DOCS = 5  # 최대 첨부 가능 문서 수


# -------------------------------
# 유틸 함수들
# -------------------------------
def load_and_split_docs(uploaded_file):
    """
    업로드된 PDF/TXT를 로컬에 저장 후 LangChain Document 리스트로 변환하고,
    RecursiveCharacterTextSplitter로 chunk 단위로 분할한다.
    """
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(uploaded_file.name)
    else:
        loader = TextLoader(uploaded_file.name, encoding="utf-8")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # 각 chunk에 source(파일명) 메타데이터 지정
    for d in docs:
        d.metadata["source"] = uploaded_file.name

    return docs


def create_vectorstore(docs):
    """
    주어진 문서들로 새 FAISS 벡터스토어를 생성하고 로컬에 저장한다.
    (이전 인덱스는 밖에서 이미 삭제했다고 가정)
    """
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(VECTORSTORE_DIR)
    return vectordb


def build_rag_chain(vectordb, task_mode: str, active_sources):
    """
    과제/레포트 도우미용 RAG 체인.
    active_sources: 사용할 문서 파일명 리스트 (None이면 전체)
    task_mode:
      - "문서 요약" / "레포트 목차 설계" / "핵심 내용 정리" / "예상 시험문제 생성" 
    """
    base_instruction = """
    너는 업로드된 문서들을 기반으로 과제와 레포트 작성을 도와주는 AI 조교야.
    항상 문서 내용을 최우선으로 참고해서 답해야 하고,
    문서에 없는 내용은 "문서에 없는 내용이라 일반적인 설명을 할게."라고 먼저 알려준 뒤 설명해.
    """

    # ### 변경 포인트 1: 
    # 더 이상 여기서 "선택 안함"을 early return 하지 않음.
    # "선택 안함"인 경우에는 아예 build_rag_chain을 호출하지 않도록 상단 로직에서 처리.

    if task_mode == "문서 요약":
        task_instruction = """
        사용자의 질문에서 지정한 범위를 중심으로 문서 내용을 3~7문장 정도로 요약해줘.
        중요 개념, 핵심 주장, 결론이 빠지지 않게 정리해.
        """
    elif task_mode == "레포트 목차 설계":
        task_instruction = """
        이 문서를 바탕으로 A4 3~5장 분량의 레포트 목차를 설계해줘.
        1, 1-1, 1-2 와 같은 계층 구조로 작성하고,
        각 소제목 옆에 한 줄씩 그 부분에 쓰면 좋을 내용을 설명해줘.
        """
    elif task_mode == "핵심 내용 정리":
        task_instruction = """
        이 문서에서 사용자가 궁금해하는 주제와 관련된 핵심 개념, 주장, 근거를
        bullet 목록 형태로 짧고 명확하게 정리해줘.
        """
    elif task_mode == "예상 시험문제 생성":
        task_instruction = """
        이 문서를 공부하는 학생에게 시험이나 구두 발표에서 나올 법한 문제를 3~5개 만들어줘.
        서술형, 객관식, 논술형 등을 섞어도 좋고,
        각 문제마다 모범답안의 핵심 포인트를 2~3줄 정도로 제시해줘.
        """
    else:  # "자유 질의응답" 등
        task_instruction = """
        사용자의 질문에 자연스럽게 답하되,
        반드시 문서에서 근거가 되는 내용 위주로 설명해줘.
        """

    prompt = ChatPromptTemplate.from_template(
        base_instruction
        + "\n\n[작업 지침]\n"
        + task_instruction
        + """
        
        [사용자 질문]
        {question}

        [참고 문서 내용]
        {context}
        """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    def get_context_from_sources(inputs):
        question = inputs["question"]
        docs = vectordb.similarity_search(question, k=12)

        # active_sources가 지정되면 해당 source만 필터링
        if active_sources:
            docs = [d for d in docs if d.metadata.get("source") in active_sources]

        if not docs:
            return "선택한 문서들에서 관련 내용을 찾지 못했습니다."

        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {
            "context": RunnableLambda(get_context_from_sources),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | prompt
        | llm
    )
    return rag_chain


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="과제·레포트 도우미 RAG 챗봇")
st.title("📚 과제·레포트 도우미 RAG 챗봇")

# 상단 소개 문구
st.markdown(
    """
**업로드한 PDF/TXT 문서를 기반으로 과제·보고서·요약·시험문제 제작을 지원하는 RAG(검색 기반 AI) 챗봇**입니다.  
아래 순서에 따라 진행하면 가장 정확한 결과를 얻을 수 있습니다.

1️⃣ **문서를 1~5개 업로드**합니다. (PDF 또는 TXT)   
2️⃣ **작업 모드를 선택**합니다.  
3️⃣ **선택한 모드에 맞게 질문 또는 요청을 입력(또는 자동 실행)**하면, 문서 내용을 중심으로 AI가 응답합니다.  

"""
)

# 세션 상태 초기화
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "sources" not in st.session_state:
    st.session_state.sources = []     # 이번 세션에서 업로드한 문서 이름들
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []   # 대화형 모드용 메시지 기록
if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = None
if "last_task_mode" not in st.session_state:
    st.session_state.last_task_mode = None
if "last_task_result" not in st.session_state:
    st.session_state.last_task_result = None

# -------------------------------
# 1️⃣ 문서 업로드
# -------------------------------
st.subheader("1️⃣ 문서 업로드 (PDF 또는 TXT)")

uploaded_files = st.file_uploader(
    "문서를 업로드하세요 (최대 5개까지 업로드 가능합니다)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    new_sources = [f.name for f in uploaded_files]

    if len(new_sources) > MAX_DOCS:
        st.error(
            f"한 번에 업로드할 수 있는 문서는 최대 {MAX_DOCS}개입니다. "
            f"현재 업로드 시도 문서 수: {len(new_sources)}개"
        )
    else:
        # 이전 인덱스 완전히 제거
        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)

        all_docs = []
        for uf in uploaded_files:
            all_docs.extend(load_and_split_docs(uf))

        st.session_state.vectordb = create_vectorstore(all_docs)
        st.session_state.sources = new_sources

        # 기존 작업 결과/상태 초기화
        st.session_state.last_task_mode = None
        st.session_state.last_task_result = None

        st.success(
            f"문서 {len(new_sources)}개를 분석 했습니다. "
        )

# 현재 세션에서 사용할 source 목록
active_sources = st.session_state.sources if st.session_state.sources else []

# -------------------------------
# 2️⃣ 작업 모드 선택
# -------------------------------
st.subheader("2️⃣ 작업 모드 선택")

mode = st.radio(
    "사용할 모드를 선택하세요.",
    ("🧩 원하는 작업을 선택하세요", "💬 자유롭게 질문해 주세요"),
    index=0,
)

is_template_mode = mode.startswith("🧩")

# 모드가 변경되면(특히 대화형 모드로 들어올 때) 채팅 기록 초기화
if st.session_state.prev_mode != mode:
    if mode.startswith("💬"):
        st.session_state.chat_messages = []
    st.session_state.prev_mode = mode

# -------------------------------
# RAG 체인 구성 (문서+모드 기반)
# -------------------------------
task_mode = None

if st.session_state.vectordb is not None and active_sources:
    if is_template_mode:
        task_mode = st.selectbox(
            "도움 받고 싶은 작업 유형을 선택하세요.",
            ["선택 안함", "문서 요약", "레포트 목차 설계", "핵심 내용 정리", "예상 시험문제 생성"],
            index=0,
        )
    else:
        # 대화형 모드: 내부적으로 '자유 질의응답'
        task_mode = "자유 질의응답"

    # ### 변경 포인트 2:
    # 작업 모드가 '선택 안함'이면 RAG 체인을 아예 만들지 않음
    if task_mode != "선택 안함":
        st.session_state.rag_chain = build_rag_chain(
            st.session_state.vectordb, task_mode, active_sources
        )
    else:
        st.session_state.rag_chain = None
else:
    st.session_state.rag_chain = None

# -------------------------------
# 3️⃣ 모드별 동작 영역
# -------------------------------
if is_template_mode:
    # -------------------------------
    # 모드 1: 작업 템플릿 모드 (드롭다운 선택 시 자동 실행)
    # -------------------------------
    # st.subheader("3️⃣ 작업선택 모드")

    # ### 변경 포인트 3:
    # 문서가 없는 경우에만 "먼저 문서를 업로드하면..." 문구를 보여줌
    if st.session_state.vectordb is None or not active_sources:
        st.info("먼저 문서를 업로드하면 작업선택 모드를 사용할 수 있습니다.")
    else:
        # 문서는 업로드된 상태
        if task_mode == "선택 안함":
            # 작업 모드를 선택하지 않은 경우: 결과 출력 X, 안내만
            st.session_state.last_task_mode = None
            st.session_state.last_task_result = None
            st.info("위 드롭다운에서 원하는 작업 유형을 선택하면, 해당 작업이 자동으로 실행됩니다.")
        elif st.session_state.rag_chain and task_mode is not None:
            # 작업 유형에 따라 자동 질문 프롬프트 구성
            if task_mode == "문서 요약":
                auto_question = "이 문서 전체를 기준으로 핵심 내용을 3~7문장으로 요약해줘."
            elif task_mode == "레포트 목차 설계":
                auto_question = "이 문서를 바탕으로 A4 3~5장 분량의 레포트 목차를 설계해줘."
            elif task_mode == "핵심 내용 정리":
                auto_question = "이 문서에서 가장 중요한 핵심 개념, 주장, 근거를 정리해줘."
            elif task_mode == "예상 시험문제 생성":
                auto_question = "이 문서로 시험이나 구두 발표에서 나올 법한 문제를 3~5개 만들어줘."
            else:  # "자유 질의응답" 등
                auto_question = "이 문서의 전체 내용을 이해하는 데 도움이 되는 핵심 설명을 해줘."

            # 작업이 변경되었을 때만 새로 실행
            if task_mode != st.session_state.last_task_mode or st.session_state.last_task_result is None:
                with st.spinner("선택한 작업을 기준으로 문서를 분석하는 중입니다..."):
                    result = st.session_state.rag_chain.invoke({"question": auto_question})
                    st.session_state.last_task_result = result.content
                    st.session_state.last_task_mode = task_mode

            # 결과 표시
            st.write(f"### ✏️ 작업 결과 – {task_mode}")
            st.write(st.session_state.last_task_result)
        else:
            # 이 경우는 거의 없지만, 안전망
            st.info("작업 유형을 다시 선택해 주세요.")
else:
    # -------------------------------
    # 모드 2: 대화형 모드 (채팅)
    # -------------------------------
    # st.subheader("3️⃣ 대화형 모드")

    # 기존 대화 내용 표시
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 채팅 입력창
    user_msg = st.chat_input("문서를 바탕으로 어떤 도움을 받고 싶나요?")

    if user_msg:
        # 문서/체인 준비 안 됐을 때
        if st.session_state.rag_chain is None:
            st.warning("먼저 문서를 업로드해야 대화형 모드를 사용할 수 있습니다.")
        else:
            # 사용자 메시지 기록 및 화면 출력
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            # RAG 호출
            with st.chat_message("assistant"):
                with st.spinner("생각 중입니다..."):
                    result = st.session_state.rag_chain.invoke({"question": user_msg})
                    answer = result.content
                    st.markdown(answer)

            # 어시스턴트 메시지 기록
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
