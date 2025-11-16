

## 🚀 프로젝트 소개

이 프로젝트는 **PDF/TXT 문서를 기반으로 학습 및 리서치를 자동화하여**,  
**요약 · 레포트 목차 설계 · 핵심 내용 정리 · 예상 시험문제 생성**을 빠르고 정확하게 만들어주는 **RAG 기반 Streamlit 웹 챗봇**입니다.

문서 기반 자동 분석뿐 아니라, **대화형 모드 + 히스토리 기억 기능**도 지원하여  
사용자와 지속적·맥락 기반 질문 응답이 가능합니다.

---

## 🧠 주요 기능 (Key Features)

| 기능 | 설명 |
|------|------|
| 📂 문서 업로드 | PDF, TXT 최대 5개 업로드 |
| 🔍 문서 기반 검색 | FAISS + Embedding 기반 Context Retrieval |
| 📝 모드 기반 자동 처리 | 요약, 목차 설계, 핵심 정리, 예상 시험 문제 |
| 💬 자유 질의응답 | 히스토리 기억 기반 AI 대화 지원 |
| 🎨 세련된 UI | Streamlit + Custom CSS 적용 |

---

## 🛠️ 기술 스택 (Tech Stack)

| Category | Tools |
|---------|----------------------------|
| Backend | Python 3.x |
| Framework | Streamlit |
| AI/LLM Engine | OpenAI GPT 모델 |
| Retrieval | FAISS Vector DB |
| Prompting | LangChain Runnables |
| Styling | Custom CSS + Gradient UI |

---


## 📦 설치 및 실행 방법

필수 패키지 설치
pip install -r requirements.txt

.env 파일 생성 후 입력:
OPENAI_API_KEY=your_api_key_here

실행
streamlit run app.py
