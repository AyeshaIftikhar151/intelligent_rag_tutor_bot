 
# 🧠 Intelligent Multi-Subject RAG Tutor Bot

An AI-powered **educational assistant** built using **LangChain, ChromaDB, and OpenAI**, designed to help students learn **English, Physics, Biology, and Pakistan Studies** with context-aware answers from uploaded study PDFs.

---

## 🚀 Features

- 🔍 **Retrieval-Augmented Generation (RAG)** for precise, context-based answers  
- 📚 Supports **multiple subjects** with individual vector databases  
- 🧠 **Memory management** for maintaining contextual conversation  
- 🧩 **Input sanitization** and **security filters**  
- 🌐 **Streamlit UI** for user-friendly chatbot interaction  
- 🔐 Keeps your **API keys safe** using `.env` and `.gitignore`

---

## 🏗️ Tech Stack

- **Python 3.10+**
- **LangChain**
- **ChromaDB**
- **HuggingFace Embeddings**
- **OpenAI / Gemini API**
- **Streamlit** (for frontend)

---

## ⚙️ Project Setup

```bash
# Clone the repository
git clone https://github.com/AyeshaIftikhar151/intelligent_rag_tutor_bot.git
cd intelligent_rag_tutor_bot

# Create virtual environment
python -m venv tutor_rag
tutor_rag\Scripts\activate   # (on Windows)

# Install dependencies
pip install -r requirements.txt
🔑 Environment Variables
Create a .env file in the project root:

bash
Copy code
GOOGLE_API_KEY=your_gemini_or_openai_key_here
CHROMA_DB_DIR=chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-4o-mini
🧩 Example .env is provided as .env.example

🖥️ Run the Chatbot (CLI)
bash
Copy code
python app.py
💬 Run the Streamlit Interface
bash
Copy code
streamlit run app_streamlit.py
🧠 Project Structure
bash
Copy code
intelligent_tutor_bot/
│
├── app.py                 # CLI chatbot
├── app_streamlit.py       # Streamlit interface
├── .env.example           # Example environment variables
├── src/                   # All utility, security, and RAG modules
├── chroma_db/             # Vector stores
├── data/                  # Uploaded PDFs
└── logs/                  # Log files
🤝 Contributing
Pull requests are welcome!
If you’d like to improve the model or add new subjects, please fork and submit a PR.

📄 License
This project is open-source under the MIT License.

yaml
Copy code
