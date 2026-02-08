# Simple RAG Chatbot

A lightweight Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Streamlit.

## Features

- 📚 Document ingestion (PDF, TXT, Markdown)
- 🔍 Vector search with ChromaDB
- 🤖 OpenAI GPT integration
- 💬 Interactive Streamlit interface
- 🎯 Context-aware responses

## Tech Stack

- **LangChain** - RAG pipeline orchestration
- **ChromaDB** - Vector database
- **OpenAI API** - Language model
- **Streamlit** - Web interface
- **Python 3.9+**

## Installation

```bash
# Clone the repository
git clone https://github.com/savinoo/simple-rag-chatbot.git
cd simple-rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

Then:
1. Upload your documents (PDF, TXT, or Markdown)
2. Ask questions about the content
3. Get accurate, context-aware answers

## How It Works

1. **Document Loading**: Upload files via Streamlit interface
2. **Text Splitting**: Documents are chunked for optimal retrieval
3. **Embedding**: Text chunks are embedded using OpenAI embeddings
4. **Vector Storage**: Embeddings stored in ChromaDB
5. **Retrieval**: User questions retrieve relevant chunks
6. **Generation**: GPT generates answers based on retrieved context

## Configuration

Edit `config.py` to customize:
- Chunk size and overlap
- Number of retrieved documents
- Model selection (GPT-3.5/GPT-4)
- Temperature and other LLM parameters

## Example Use Cases

- Internal knowledge base search
- Customer support documentation
- Research paper Q&A
- Technical documentation assistant

## Project Structure

```
simple-rag-chatbot/
├── app.py              # Streamlit application
├── rag_pipeline.py     # RAG implementation
├── config.py           # Configuration
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

## License

MIT

## Author

Lucas Lorenzo Savino  
AI Engineer | Agent Development & MLOps

---

*Part of my AI engineering portfolio demonstrating RAG implementation skills.*
