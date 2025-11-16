import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 1. 환경 변수 로드 (.env 파일 안에 OpenAI API 키가 저장되어 있음)
load_dotenv(".env")

# 2. 벡터스토어(임베딩 데이터베이스) 저장 폴더 설정
VECTORSTORE_DIR = "faiss_index"

# 3. 문서 로드 및 텍스트 분할 함수
def load_and_split_docs(uploaded_file):
    """
    사용자가 업로드한 PDF 또는 TXT 문서를 읽고
    LangChain에서 처리 가능한 문서 객체 리스트로 변환하는 함수.
    - PDF: PyPDFLoader 사용
    - TXT: TextLoader 사용
    이후 RecursiveCharacterTextSplitter를 이용해 일정 단위로 분할한다.
    """
    # 업로드한 파일을 임시로 로컬에 저장
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 파일 확장자에 따라 다른 로더 선택
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(uploaded_file.name)
    else:
        loader = TextLoader(uploaded_file.name, encoding="utf-8")

    # 문서 로드 (LangChain Document 객체로 반환됨)
    documents = loader.load()

    # 문서를 500자 단위로 나누고, 100자 중첩(Overlapping) 적용
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)


# 4. 벡터스토어 생성 함수 (새 문서 업로드 시 최초 1회 실행)
def create_vectorstore(docs):
    """
    분할된 문서들을 OpenAI 임베딩으로 벡터화한 후,
    FAISS(Vector Store)에 저장하는 함수.
    이후 검색을 빠르게 하기 위해 로컬에 저장한다.
    """
    embeddings = OpenAIEmbeddings()                    # OpenAI 임베딩 모델 초기화
    vectordb = FAISS.from_documents(docs, embeddings)  # 문서 임베딩 → 벡터 인덱스 생성
    vectordb.save_local(VECTORSTORE_DIR)               # 로컬 폴더에 저장
    return vectordb


# 5. 기존 벡터스토어 로드 함수
def load_vectorstore():
    """
    이미 만들어진 FAISS 인덱스가 로컬에 존재할 경우 이를 불러오는 함수.
    존재하지 않거나 오류가 있으면 None 반환.
    """
    embeddings = OpenAIEmbeddings()
    if os.path.exists(VECTORSTORE_DIR):
        try:
            # langchain 버전에 따라 allow_dangerous_deserialization=True 옵션이
            # 필요할 수 있음. 문제 생기면 아래 줄처럼 수정:
            return FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
            #return FAISS.load_local(VECTORSTORE_DIR, embeddings)
        except Exception as e:
            st.warning(f"벡터스토어 로드 중 오류 발생: {e}")
            return None
    return None


# 6. RAG (Retrieval-Augmented Generation) 체인 구성 함수
def build_rag_chain(vectordb, task_mode: str):
    """
    과제/레포트 도우미용 RAG 체인.
    - retriever: 사용자의 질문과 유사한 문서 조각 검색
    - prompt: 작업 유형(task_mode)에 따라 다른 지시를 포함
    - llm: ChatOpenAI가 최종 답변 생성
    """
    retriever = vectordb.as_retriever()

    # 공통 시스템 지침
    base_instruction = """
    너는 업로드된 문서를 기반으로 과제와 레포트 작성을 도와주는 AI 조교야.
    항상 문서 내용을 최우선으로 참고해서 답해야 하고,
    문서에 없는 내용은 "문서에 없는 내용이라 일반적인 설명을 할게."라고 먼저 알려준 뒤 설명해.
    """

    # 작업 유형별 추가 지침
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

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # retriever 결과(Document 리스트)를 텍스트로 합쳐주는 람다
    def join_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) 
                       | retriever 
                       | RunnableLambda(lambda docs: join_docs(docs)),
            "question": RunnableLambda(lambda x: x["question"])
        }
        | prompt
        | llm
    )
    return rag_chain


# 7. Streamlit 웹 인터페이스 설정
st.set_page_config(page_title="과제·레포트 도우미 RAG 챗봇")
st.title("📚 과제·레포트 도우미 RAG 챗봇")

st.write("문서를 업로드한 뒤, 왼쪽에서 작업 유형을 선택하고 질문을 입력하면 됩니다.")

# 🔧 사이드바: 작업 유형 선택
st.sidebar.header("작업 유형 설정")
task_mode = st.sidebar.selectbox(
    "어떤 도움을 받고 싶나요?",
    ["문서 요약", "레포트 목차 설계", "핵심 내용 정리", "예상 시험문제 생성", "자유 질의응답"],
    index=0
)

# 8. 세션 상태 초기화
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 9. 로컬에 벡터스토어가 이미 존재하는지 확인
vectordb_exists = os.path.exists(VECTORSTORE_DIR)

# 10. 문서 업로드 UI (PDF, TXT 파일 허용)
uploaded_file = st.file_uploader("문서를 업로드하세요 (PDF 또는 TXT)", type=["pdf", "txt"])

# 11. 벡터스토어 존재 시: 로드 후 바로 사용
if vectordb_exists and st.session_state.vectordb is None:
    st.session_state.vectordb = load_vectorstore()
    if st.session_state.vectordb:
        st.session_state.rag_chain = build_rag_chain(st.session_state.vectordb, task_mode)
        st.success("기존 벡터스토어를 불러왔습니다.")
    else:
        st.warning("벡터스토어를 불러오지 못했습니다. 새로 생성하세요.")

# 12. 벡터스토어가 없을 때: 업로드된 문서로 새로 생성
elif not vectordb_exists:
    if uploaded_file:
        with st.spinner("문서를 처리하고 임베딩 중입니다..."):
            split_docs = load_and_split_docs(uploaded_file)              # 문서 로드 및 분할
            st.session_state.vectordb = create_vectorstore(split_docs)   # 벡터스토어 생성
            st.session_state.rag_chain = build_rag_chain(
                st.session_state.vectordb, task_mode
            )
            st.success("새 벡터스토어를 생성했습니다.")
    else:
        st.info("벡터스토어가 없으므로 문서를 업로드해야 합니다.")

# 🔄 작업 유형이 바뀌면 체인을 다시 구성
if st.session_state.vectordb is not None:
    st.session_state.rag_chain = build_rag_chain(
        st.session_state.vectordb, task_mode
    )

# 13. 사용자 질의 입력 및 답변 출력
if st.session_state.rag_chain:
    # 질문 입력
    question = st.text_area(
        "질문이나 원하는 작업 범위를 입력하세요.\n예) 2~3페이지 중심으로 요약해줘 / 환경오염 파트만 레포트 목차 짜줘 등",
        height=120
    )

    # ✅ 쿼리 실행 버튼
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
    st.info("먼저 문서를 업로드하고, 벡터스토어를 생성/로드해야 합니다.")
