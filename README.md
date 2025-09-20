
# 🛡️ TruthGuard

**TruthGuard** is a prototype for an AI-powered misinformation detection platform.  
It helps users analyze text and URLs to identify potential misinformation, deepfakes, and biased content using advanced machine learning models.

---

## ✨ Features

- **Text Analysis:** Detects patterns of misinformation in text-based content.  
- **URL Analysis:** Analyzes URLs to identify potential phishing or malicious threats.  
- **AI Capabilities:** Showcases a range of advanced AI features like Deepfake Detection, Bias Detection, and Source Credibility Scoring.  
- **Educational Resources:** Provides insights and resources to help users become more critical consumers of digital information.  
- **Interactive UI:** A sleek, modern, and responsive interface designed with a cyber-tech aesthetic.  

---

## 💻 Tech Stack

This project is a full-stack application composed of a **React frontend** and a **Python backend** that handles AI model inference.

### Frontend
- ⚛️ **React** – User interface  
- 🟦 **TypeScript** – Type safety  
- 🎨 **Tailwind CSS** – Utility-first styling  
- 🧩 **Shadcn/ui** – Accessible, customizable components  
- ⚡ **Vite** – Build tool for fast development  

### Backend
- 🚀 **FastAPI** – High-performance Python web framework  
- 🤗 **Transformers** – Hugging Face library for pre-trained models  
- 🔥 **PyTorch** – Deep learning framework  

---

## 🚀 Getting Started

To run this project, set up both the **backend** and the **frontend**.

### 1️⃣ Backend Setup

The backend handles AI analysis and serves as the API for the frontend.

1. Navigate to the backend directory (e.g., `truthguard-backend`).  
2. Install dependencies:  
   ```bash
   pip install "fastapi[all]" "uvicorn[standard]" transformers torch
````

3. Start the FastAPI server (models will download on first run):

   ```bash
   uvicorn app:app --reload
   ```

   API runs at: `http://localhost:8000`

---

### 2️⃣ Frontend Setup

1. Navigate to the frontend root directory.
2. Install packages:

   ```bash
   npm install
   # or
   yarn install
   # or
   bun install
   ```
3. Start the development server:

   ```bash
   npm run dev
   # or
   yarn dev
   # or
   bun dev
   ```

Frontend runs at: `http://localhost:5173`
⚠️ Make sure the backend is running first — otherwise the **Content Analyzer** won’t work.

---

## 🤝 Contributing

Contributions are welcome! 🎉
If you have ideas for new features or improvements, feel free to open an **issue** or submit a **pull request**.

---

## 📄 License

This project is licensed under the **MIT License**.

```

Do you want me to also add **badges** (like build status, license, tech stack logos) at the top to give it a more professional GitHub look?
```
