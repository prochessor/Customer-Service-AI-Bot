import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables from .env
load_dotenv()

# Initialize Groq model
llm = ChatGroq(
    temperature=0.8,
    model="llama3-70b-8192",
    api_key="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"
)

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings()

# File path for saving vector database
vectordb_file_path = "faiss_index"

# Function to create vector database from CSV
def create_vector_db(csv_file):
    loader = CSVLoader(file_path=csv_file, source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)
    return vectordb

# Function to get QA chain
def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """While providing the answer don't say "According to the context...' or something like that go straight into the answer.

    CONTEXT: {context}

    QUESTION: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

# Streamlit UI
def main():
    st.title("🤖 CSV-based Customer Service Bot")
    st.write("Upload a CSV file containing FAQs and get answers to your questions based on the content.")
    st.write("Use the input box below to ask questions.")

    # Sidebar for admin information
    st.sidebar.title("Admin Information")
    st.sidebar.title("ℹ️ CSV File Requirements")
    st.sidebar.write("Upload a CSV file containing FAQs with two columns:")
    st.sidebar.write("1. 'prompt': Contains the question or prompt.")
    st.sidebar.write("2. 'response': Contains the corresponding answer.")

    st.sidebar.write("---")

    st.sidebar.write("After uploading the CSV file, click on the button to create the vector database.")

    # Upload CSV and create vector database
    with st.sidebar.expander("Click Here to proceed"):
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            csv_filename = uploaded_file.name
            with open(csv_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"CSV file '{csv_filename}' uploaded successfully!")

            # Create vector database button
            if st.button("Create Vector Database"):
                vectordb = create_vector_db(csv_filename)
                st.session_state.vectordb = vectordb  # Store vector database in session state
                st.success("Vector database created successfully!")

    # Check if vector database exists in session state
    if "vectordb" in st.session_state:
        question = st.text_input("Ask a question:")
        if question:
            chain = get_qa_chain(st.session_state.vectordb)
            result = chain.invoke(question)
            st.write("**Answer:**")
            st.write(result["result"])

            # Show relevant source documents in dropdown
            if result["source_documents"]:
                st.write("**Relevant Questions:**")
                for doc in result["source_documents"]:
                    # Ensure 'metadata' key exists before accessing 'prompt'
                    with st.expander(doc.metadata["source"]):
                        content = doc.page_content
                        response = content.split("response:")[1].strip() if "response:" in content else "No response found"
                        st.write(response)
    else:
        st.warning("Please upload a CSV and create a vector database to proceed.")

    st.write("💬 Feel free to ask any question you have!")

if __name__ == "__main__":
    main()
