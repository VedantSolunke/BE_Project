# Legal Vault - Indian Penal Code Assistant

## Overview
Legal Vault is an AI-powered legal assistance platform that helps users find relevant Indian Penal Code (IPC) sections based on their case descriptions. The application uses advanced natural language processing, OpenAI embeddings, and FAISS vector search to match case descriptions with the most relevant IPC sections, providing detailed legal analysis.

## Features
- 🔍 **Smart Search**: Find relevant IPC sections based on natural language case descriptions
- 📊 **Relevance Rankings**: Sections are ranked and color-coded by relevance
- 🧠 **AI Legal Analysis**: Get AI-powered case-specific legal insights
- 📑 **Detailed Information**: Access comprehensive details about each IPC section
- 🎨 **User-Friendly Interface**: Interactive and visually organized results

## Technologies Used
- **Streamlit**: For the web interface
- **OpenAI API**: For embeddings and GPT-based legal analysis
- **FAISS**: For efficient vector similarity search
- **LangChain**: For LLM orchestration and tracking
- **LangSmith**: For monitoring and debugging AI interactions

## Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API key
- LangSmith API key (optional, for tracing)

### Installation
1. Clone this repository
```bash
git clone https://github.com/yourusername/legal-vault.git
cd legal-vault
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys
```
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name
LANGSMITH_TRACING=true
```

4. Run the application
```bash
./run_app.sh
# Or directly with: streamlit run app.py
```

## Deployment
This application can be deployed on platforms like Streamlit Cloud, Hugging Face Spaces, or any platform that supports Python web applications. For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## License
MIT

## Acknowledgments
- Thanks to all contributors who helped develop this legal assistance platform
- Indian Penal Code dataset curated from official legal resources
