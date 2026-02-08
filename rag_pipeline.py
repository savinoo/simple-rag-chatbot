"""
RAG Pipeline Implementation
Author: Lucas Lorenzo Savino
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
import tempfile
import os
import config


class RAGPipeline:
    """RAG pipeline for document Q&A"""
    
    def __init__(self, api_key):
        """Initialize RAG pipeline
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectorstore = None
        self.qa_chain = None
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len
        )
    
    def load_documents(self, uploaded_files):
        """Load and process uploaded documents
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
        """
        documents = []
        
        for uploaded_file in uploaded_files:
            # Save to temp file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Load based on file type
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                elif uploaded_file.name.endswith('.txt'):
                    loader = TextLoader(tmp_path)
                elif uploaded_file.name.endswith('.md'):
                    loader = UnstructuredMarkdownLoader(tmp_path)
                else:
                    continue
                
                docs = loader.load()
                documents.extend(docs)
            finally:
                # Cleanup temp file
                os.unlink(tmp_path)
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=None  # In-memory
        )
        
        # Create QA chain
        llm = ChatOpenAI(
            model_name=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            openai_api_key=self.api_key
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": config.K_DOCUMENTS}
            ),
            return_source_documents=True
        )
    
    def query(self, question, temperature=None, k=None):
        """Query the RAG system
        
        Args:
            question: User question
            temperature: LLM temperature (overrides default)
            k: Number of documents to retrieve (overrides default)
            
        Returns:
            dict: Answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("No documents loaded. Please upload documents first.")
        
        # Update parameters if provided
        if temperature is not None or k is not None:
            llm_kwargs = {}
            if temperature is not None:
                llm_kwargs['temperature'] = temperature
            
            retriever_kwargs = {}
            if k is not None:
                retriever_kwargs['search_kwargs'] = {"k": k}
            
            if llm_kwargs:
                self.qa_chain.combine_documents_chain.llm_chain.llm.temperature = temperature
            
            if retriever_kwargs:
                self.qa_chain.retriever.search_kwargs["k"] = k
        
        # Run query
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }
