# 🧠 RAGnosis — Chat with Your Documents

RAGnosis is an AI-powered application that lets you chat with your own documents.  
It uses Retrieval-Augmented Generation (RAG) to provide accurate, context-based answers from local PDFs.

---

## 🚀 Features

- 📄 Chat with local PDF documents  
- 🔍 Semantic search using embeddings  
- ⚡ Fast retrieval using Chroma vector database  
- 🤖 Context-aware answers with Gemini LLM  
- 🌐 Simple and clean UI using Gradio  
- 🧠 Reduces hallucination by grounding answers in document context  

---

🧠 How It Works
-📄 Load Document
The PDF file is loaded using LangChain document loader
-✂️ Text Splitting
The document is split into smaller chunks for better processing
-🔢 Embedding Creation
Each chunk is converted into vector embeddings using Hugging Face model
-🗂 Vector Storage
Embeddings are stored in Chroma vector database
-🔍 Retrieval
When a query is asked, the system retrieves the most relevant chunks
-🤖 Answer Generation
The retrieved context is passed to the Gemini model to generate accurate answers

---

## 🛠️ Tech Stack

- Python  
- LangChain  
- ChromaDB  
- Hugging Face Embeddings  
- Google Gemini API  
- Gradio  

---

## 📸 Demo
Screenshot given in the repository
Use your own api keys

---

## 📁 Project Structure

RAGnosis/
│── app.py
│── attention-is-all-you-need.pdf
│── requirements.txt
│── .env 

---

##⚙️ Installation

git clone https://github.com/your-bharath-kumar-macharla/RAGnosis.git
cd Ragnosis
pip install -r requirements.txt

---

##▶️ How to Run the Application
python app.py
After running, open your browser and go to:
http://127.0.0.1:7860

---

##🔥 Future Improvements
📄 Upload custom PDFs
💬 Chat history (like ChatGPT)
🌐 Deploy to cloud
📊 Show sources in UI

---

##🤝 Contributing

Feel free to fork and improve this project.


