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
    """
    base_instruction = """
    너는 업로드된 문서들을 기반으로 과제와 레포트 작성을 도와주는 AI 조교야.
    항상 문서 내용을 최우선으로 참고해서 답해야 하고,
    문서에 없는 내용은 "문서에 없는 내용이라 일반적인 설명을 할게."라고 먼저 알려준 뒤 설명해.
    """

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
    else:  # "자유 질의응답"
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

# 세션 상태 초기화
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "sources" not in st.session_state:
    st.session_state.sources = []     # 이번 세션에서 업로드한 문서 이름들
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 1) (기존 사이드바 → 메인 상단) 작업 유형 선택
st.subheader("1️⃣ 작업 유형 선택")

task_mode = st.selectbox(
    "도움 받고 싶은 작업 유형을 먼저 선택하세요! 응답의 품질을 높일 수 있습니다.",
    ["문서 요약", "레포트 목차 설계", "핵심 내용 정리", "예상 시험문제 생성", "자유 질의응답"],
    index=0,
)

# 2) 문서 업로드 (여러 개, 최대 5개)
st.subheader("2️⃣ 문서 업로드 (PDF 또는 TXT, 여러 개 가능)")

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

        st.success(
            f"문서 {len(new_sources)}개를 새 벡터스토어로 생성했습니다. "
            "(이전 문서들은 더 이상 참조하지 않습니다.)"
        )

# 현재 세션에서 사용할 source 목록
active_sources = st.session_state.sources if st.session_state.sources else []

# 3) RAG 체인 구성
if st.session_state.vectordb is not None and active_sources:
    st.session_state.rag_chain = build_rag_chain(
        st.session_state.vectordb, task_mode, active_sources
    )
else:
    st.session_state.rag_chain = None

# 4) 질의 입력 + 버튼으로 쿼리 실행
st.subheader("3️⃣ 질문 입력 및 실행")

if st.session_state.rag_chain:
    question = st.text_area(
        "질문이나 원하는 작업 범위를 입력하세요.\n"
        "예) 2~3페이지 중심으로 요약해줘 / 환경오염 파트만 레포트 목차 짜줘 / "
        "이 문서로 시험에 나올 법한 문제 5개 만들어줘 등",
        height=150,
    )

    run_query = st.button("쿼리 실행")

    if run_query:
        if question.strip():
            with st.spinner("과제/레포트 도와주는 중..."):
                result = st.session_state.rag_chain.invoke({"question": question})
                st.write("### ✏️ 결과")
                st.write(result.content)
        else:
            st.warning("먼저 질문을 입력해주세요.")
else:
    st.info("먼저 문서를 업로드하면 사용할 수 있습니다.")
