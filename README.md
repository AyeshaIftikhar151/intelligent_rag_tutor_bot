# 🧠 Intelligent Tutor RAG Chatbot

An **AI-powered educational tutor** that provides intelligent, context-aware answers across multiple subjects — including English, Physics, Biology, and Pakistan Studies.  
Built using **LangChain, OpenAI, ChromaDB, and Streamlit**.

---

## 🚀 Features

- 🧩 **Multi-subject RAG** (Retrieval-Augmented Generation) chatbot  
- 🔒 **Secure input handling** with sanitization  
- 🧠 **Contextual memory** (subject-based memory storage)  
- 📚 **PDF ingestion** for each subject (English, Physics, Biology, Pakistan Studies)  
- 💬 **Interactive Streamlit UI**  
- ⚙️ **Hybrid document retrieval** with re-ranking  
- 🔍 **User-friendly name memory and small talk recognition**

---

## 🏗️ Project Structure

intelligent_tutor_bot/
│
├── app_streamlit.py # Streamlit front-end interface
├── src/
│ ├── utils/ # Config, guardrails, sanitizer, memory manager
│ ├── rag/ # RAG retriever logic
│ ├── secuirity/ # Input sanitization & safety
│ ├── logger.py # Centralized logging
│ └── ...
│
├── chroma_db/ # Vector database for each subject
├── data/ # Subject PDFs (English, Physics, Bio, Pak Studies)
├── .env # Environment variables
├── .gitignore
└── README.md

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/intelligent_tutor_rag_bot.git
cd intelligent_tutor_rag_bot
2️⃣ Create virtual environment
bash
Copy code
python -m venv lanchain_env
source lanchain_env/bin/activate   # (Linux/macOS)
lanchain_env\Scripts\activate      # (Windows)
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Add environment variables
Create a .env file in the project root:

bash
Copy code
OPENAI_API_KEY=your_api_key_here
CHROMA_DB_DIR=chroma_db
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
🧠 Running the App
Start the Streamlit app:

bash
Copy code
streamlit run app_streamlit.py
Then open your browser at
👉 http://localhost:8501

🧩 Subjects Supported
Subject	PDF Filename	Description
English	english.pdf	Grammar, writing, comprehension
Physics	physics_notes.pdf	Motion, energy, laws, and forces
Biology	bio.pdf	Cells, proteins, genetics
Pakistan Studies	pakstudies_dates.pdf	History, independence, geography

🧰 Technologies Used
LangChain

ChromaDB

OpenAI GPT / Gemini

Sentence Transformers

Streamlit

dotenv

Python 3.10+

🛡️ Security
Input sanitization using a custom Sanitizer

Guardrails for detecting small talk and name extraction

Isolated memory management for each subject

Secure handling of .env and local PDFs

📜 License
This project is released under the MIT License.
Feel free to fork and extend the project for your academic or research use.

💡 Built by Ayesha — making AI education intelligent and secure!