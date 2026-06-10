# 🤖 AI/ML Code Generation Agent

A **multi-agent AI system** that takes a plain-English ML task description and automatically generates a complete, reviewed Python script — powered by **CrewAI**, **LangChain**, and **Google Gemini**.

---

## 🧠 How It Works

The pipeline runs three specialised agents sequentially:

```
User Prompt → 🗺️ Planner → 💻 Coder → 🔍 Reviewer → Final Code
```

| Agent | Role |
|-------|------|
| 🗺️ **Planner** | Analyses the ML task and designs a structured blueprint (problem type, preprocessing, model choice, evaluation strategy) |
| 💻 **Coder** | Translates the blueprint into clean, well-commented, runnable Python code |
| 🔍 **Reviewer** | Audits the code for bugs, inefficiencies, and PEP-8 violations — returns corrected code with a review summary |

---

## 🚀 Demo

![App Screenshot](assets/screenshot.png)

**Example prompts:**
- `Binary classification on the Titanic dataset using XGBoost`
- `House price prediction with linear regression and feature engineering`
- `Sentiment analysis on tweets using a fine-tuned BERT model`

---

## 🛠️ Tech Stack

- **[CrewAI](https://github.com/joaomdmoura/crewAI)** — multi-agent orchestration
- **[LangChain](https://github.com/langchain-ai/langchain)** — LLM tooling and chaining
- **[Google Gemini 1.5 Pro](https://ai.google.dev/)** — LLM backbone
- **[Streamlit](https://streamlit.io/)** — interactive web UI

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-ml-code-gen-agent.git
cd ai-ml-code-gen-agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and add your **Google Gemini API key**:
```
GOOGLE_API_KEY=your_google_api_key_here
```
Get a free key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 5. Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
ai-ml-code-gen-agent/
│
├── agents.py          # Agent definitions + CrewAI pipeline
├── app.py             # Streamlit UI
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
├── .gitignore
└── README.md
```

---

## 📌 Key Features

- ✅ Multi-agent workflow (Planner → Coder → Reviewer)
- ✅ Supports classification, regression, clustering, NLP, and CV tasks
- ✅ Generates complete, runnable Python scripts
- ✅ Self-correcting code via the Reviewer agent
- ✅ One-click `.py` file download
- ✅ Example prompts in the sidebar

---

## 🔑 API Keys Required

| Service | Where to get it |
|---------|----------------|
| Google Gemini | [aistudio.google.com](https://aistudio.google.com/app/apikey) |

---

## 👤 Author

**Mohammed Jawad**  
AI Engineer & Data Scientist  
📧 mdjawad9228@gmail.com  
🔗 [LinkedIn](https://linkedin.com) · [GitHub](https://github.com)

---

## 📄 License

MIT License — free to use and modify.
