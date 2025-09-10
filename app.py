import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Optional PEFT import for LoRA-based models
try:
    from peft import PeftModel, PeftConfig
    peft_available = True
except ImportError:
    peft_available = False

# Set page config
st.set_page_config(
    page_title="Nexo AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# App title and description
st.title("🤖 Nexo - AI Assistant")
st.markdown("Created by **Aayush Bohora**")
st.markdown("---")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar info (no model selector now)
with st.sidebar:
    st.header("⚙️ Settings")
    
    max_length = st.slider("Response Length", 50, 300, 150)
    temperature = st.slider("Creativity", 0.1, 1.0, 0.7)
    
    st.markdown("---")
    st.header("ℹ️ About Nexo")
    st.info("""
    **Nexo** is an AI assistant created by **Aayush Bohora**. 
    It's powered by the fine-tuned `dolly_chatbot_model`.
    """)
    
    st.markdown("---")
    st.caption("Version 1.1 • Built with Streamlit")

# Function to clean model responses
def clean_response(response):
    response = re.sub(r'[^.!?]+$', '', response)
    response = re.sub(r'\*\*|\*|__|_', '', response)
    if response.startswith('1.') and len(response.split('\n')) < 3:
        lines = response.split('\n')
        if len(lines) > 0:
            response = lines[0].replace('1. ', '', 1)
    response = response.strip()
    return response

# Function to load model
@st.cache_resource(show_spinner=False)
def load_model(model_name="dolly_chatbot_model"):
    try:
        with st.spinner(f"Loading {model_name}... This may take a minute."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if peft_available:
                try:
                    config = PeftConfig.from_pretrained(model_name)
                    base_model_name = config.base_model_name_or_path
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                    model = PeftModel.from_pretrained(base_model, model_name)
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None, None

# Load the dolly model once
if not st.session_state.model_loaded:
    st.session_state.tokenizer, st.session_state.model = load_model("dolly_chatbot_model")
    st.session_state.model_loaded = st.session_state.tokenizer is not None and st.session_state.model is not None
    if st.session_state.model_loaded:
        st.success("✅ dolly_chatbot_model loaded successfully!")
    else:
        st.error("❌ Failed to load dolly_chatbot_model")

# Function to generate response
def generate_response(user_input, max_length=150, temperature=0.7):
    if not st.session_state.model_loaded:
        return "Model not loaded yet. Please wait..."
    try:
        if user_input.lower().startswith(('explain', 'what is', 'what are', 'how', 'why')):
            prompt = f"Please provide a clear and concise explanation for: {user_input}\n\nResponse:"
        elif user_input.lower().startswith(('write', 'create', 'generate')):
            prompt = f"Please create the following: {user_input}\n\nResponse:"
        elif user_input.lower().startswith(('list', 'name some', 'give examples')):
            prompt = f"Please provide a list for: {user_input}\n\nResponse:"
        else:
            prompt = f"Please respond to the following: {user_input}\n\nResponse:"
        
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=st.session_state.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Response:" in response:
            response = response.split("Response:")[1].strip()
        return clean_response(response)
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main chat interface
st.header("💬 Chat with Nexo")
user_input = st.text_area(
    "Enter your question or instruction:",
    placeholder="e.g., Explain machine learning in simple terms\nor\nWrite a poem about AI\nor\nHow do I make chocolate chip cookies?",
    height=120,
    key="user_input"
)

if st.button("Get Response", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("Nexo is thinking..."):
            response = generate_response(user_input, max_length, temperature)
            st.subheader("Nexo's Response:")
            st.success(response)
    else:
        st.warning("Please enter a question or instruction first.")

# Example prompts
with st.expander("💡 Example Questions to Ask"):
    st.markdown("""
    **For best results, try these types of questions:**
    
    - **Explanation**: "Explain quantum computing in simple terms"
    - **Creative**: "Write a short story about a robot who becomes human"
    - **Instructional**: "How do I learn Python programming?"
    - **Comparative**: "What's the difference between AI and machine learning?"
    - **Definition**: "What is blockchain technology?"
    - **List-based**: "List 5 benefits of renewable energy"
    """)

# Footer
st.markdown("---")
st.caption("© 2024 Nexo AI Assistant • Created by Aayush Bohora")
