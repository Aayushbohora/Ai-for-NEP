# 🤖 Nexo AI Assistant (Dolly Chatbot Model)

This project contains **Nexo AI Assistant**, a lightweight chatbot model created by **Aayush Bohora**.  
It is built on top of a **small language model** and fine-tuned for **chat-style interactions**.  

---

## 📌 Model Details
- **Base Model:** Dolly (small variant)  
- **Parameters:** ~125M (low-parameter model, suitable for lightweight tasks)  
- **Fine-tuning:** Trained on a **chat dataset** for question-answer and dialogue tasks  
- **Frameworks:** Hugging Face Transformers, PyTorch  

---

## ⚡ Why This Model?
This model is intentionally **small and lightweight**, making it easier to run on limited hardware (CPU or low VRAM GPUs).  
However, because of its small size, the responses may sometimes be **incomplete or less accurate** compared to larger LLMs.  

---

## 🚀 Future Improvements
It would be great if the community helps improve this model by:
- Fine-tuning it further on **larger, high-quality chat datasets**  
- Applying **LoRA / PEFT techniques** for better alignment  
- Scaling up to **bigger Dolly or GPT-Neo variants** while keeping efficiency in mind  
 if you want to try https://nexoai.streamlit.app/
---

## 🛠️ Running the Assistant
You can launch the chatbot with **Streamlit**:

```bash
pip install -r requirements.txt
streamlit run app.py
