import streamlit as st
import logging
import ollama

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any
from PyPDF2 import PdfReader

st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=True)
def extract_model_names(
        models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names


def get_pdf_texts(pdf_docs):
    texts = ''

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            texts += page.extract_text()

    return texts


def create_vector_db(text) -> Chroma:
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=7500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logger.info("Document split into chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    vector_db = Chroma.from_texts(
        texts=chunks, embedding=embeddings, collection_name="RAG"
    )

    return vector_db


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    logger.info(f"""Processing question: {question} using model: {selected_model}""")
    llm = ChatOllama(model=selected_model, temperature=0)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Bạn là trợ lý mô hình ngôn ngữ AI. Nhiệm vụ của bạn là tạo ra 3
         các phiên bản khác nhau của câu hỏi người dùng nhất định để truy xuất các 
         tài liệu liên quan từ vector database. Bằng cách tạo ra nhiều góc nhìn cho 
         câu hỏi của người dùng. Mục tiêu là giúp người dùng khắc phục một số hạn chế 
         của việc truyền dữ liệu dựa trên khoảng cách tìm kiếm tương tự. Cung cấp các 
         câu hỏi thay thế này được phân tách bằng dòng mới.
         Câu hỏi gốc: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Trả lời câu hỏi dựa theo nội dung của tài liệu sau:
    {context}
    Câu hỏi: {question}
    Nếu bạn không biết vui lòng trả lời một cách lịch sử là tôi không biết, không được phép tự ý bịa đặt 
    Chỉ trả lời các câu hỏi ở trong {context}, không trả lời ở ngoài nội dung file.
    Thêm đoạn ngữ cảnh bạn đã sử dụng để trả lời câu hỏi.
    Chỉ trả lời bằng tiếng việt
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    response = chain.invoke(question)
    return response


def main() -> None:
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    if available_models:
        selected_model = col2.selectbox(
            "Chọn model ↓", available_models
        )

    file_upload = col1.file_uploader(
        "Chọn file ↓", type="pdf", accept_multiple_files=True
    )

    if file_upload:
        st.session_state["file_upload"] = file_upload
        if st.session_state["vector_db"] is None:
            chunk_texts = get_pdf_texts(file_upload)
            st.session_state["vector_db"] = create_vector_db(chunk_texts)

    with col2:
        message_container = st.container(height=300, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Nhập vào nội dung..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[Processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Vui lòng chọn file.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Chọn file để bắt đầu...")


if __name__ == "__main__":
    main()
