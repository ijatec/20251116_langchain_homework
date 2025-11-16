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
    embeddings = OpenAIEmbeddings()                   # OpenAI 임베딩 모델 초기화
    vectordb = FAISS.from_documents(docs, embeddings) # 문서 임베딩 → 벡터 인덱스 생성
    vectordb.save_local(VECTORSTORE_DIR)              # 로컬 폴더에 저장
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
            return FAISS.load_local(VECTORSTORE_DIR, embeddings)
        except Exception as e:
            st.warning(f"벡터스토어 로드 중 오류 발생: {e}")
            return None
    return None


# 6. RAG (Retrieval-Augmented Generation) 체인 구성 함수
def build_rag_chain(vectordb):
    """
    RAG 체인은 '검색 + 생성'을 결합한 구조.
    - retriever: 사용자의 질문과 유사한 문서 조각 검색
    - prompt: 검색된 문맥(context)을 포함하여 모델에 질의
    - llm: ChatOpenAI 모델이 최종 답변 생성
    """
    retriever = vectordb.as_retriever()  # 벡터스토어 → retriever 객체로 변환

    # 답변 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_template(
        """
        너는 문서를 기반으로 답변하는 AI야.
        주어진 문서를 참고해 아래 질문에 정확하고 간결하게 답해:

        질문: {question}

        참고 문서:
        {context}
        """
    )

    # ChatOpenAI 모델 호출 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Runnable 체인 구성:
    # 사용자의 질문을 retriever에 전달하여 context를 가져오고,
    # 그 결과를 prompt와 LLM으로 이어붙이는 파이프라인 형태로 연결
    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever,
            "question": RunnableLambda(lambda x: x["question"])
        }
        | prompt
        | llm
    )
    return rag_chain


# 7. Streamlit 웹 인터페이스 설정
st.set_page_config(page_title="문서 RAG 챗봇")
st.title("문서 요약 및 질의응답 챗봇")

# 8. 세션 상태 초기화
# Streamlit은 앱을 새로고침해도 상태를 기억하기 위해 session_state 사용
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 9. 로컬에 벡터스토어가 이미 존재하는지 확인
vectordb_exists = os.path.exists(VECTORSTORE_DIR)

# 10. 문서 업로드 UI (PDF, TXT 파일 허용)
uploaded_file = st.file_uploader("문서를 업로드하세요 (PDF 또는 TXT)", type=["pdf", "txt"])

# 11. 벡터스토어 존재 시: 로드 후 바로 사용
if vectordb_exists:
    st.session_state.vectordb = load_vectorstore()
    if st.session_state.vectordb:
        st.session_state.rag_chain = build_rag_chain(st.session_state.vectordb)
        st.success("기존 벡터스토어를 불러왔습니다.")
    else:
        st.warning("벡터스토어를 불러오지 못했습니다. 새로 생성하세요.")

# 12. 벡터스토어가 없을 때: 업로드된 문서로 새로 생성
else:
    if uploaded_file:
        with st.spinner("문서를 처리하고 임베딩 중입니다..."):
            split_docs = load_and_split_docs(uploaded_file)           # 문서 로드 및 분할
            st.session_state.vectordb = create_vectorstore(split_docs) # 벡터스토어 생성
            st.session_state.rag_chain = build_rag_chain(st.session_state.vectordb)
            st.success("새 벡터스토어를 생성했습니다.")
    else:
        st.info("벡터스토어가 없으므로 문서를 업로드해야 합니다.")

# 13. 사용자 질의 입력 및 답변 출력
if st.session_state.rag_chain:
    # 사용자로부터 질문 입력받기
    question = st.text_input("질문을 입력하세요:")

    # 질문이 입력되면 RAG 체인 실행
    if question:
        with st.spinner("답변 생성 중..."):
            # RAG 파이프라인 실행: {"question": 질문} 형태로 입력
            result = st.session_state.rag_chain.invoke({"question": question})
            st.write("### 답변:")
            st.write(result.content)
