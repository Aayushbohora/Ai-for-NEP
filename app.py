import streamlit as st
import sys
import subprocess
import os

# Function to install missing packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install required packages
required_packages = [
    "streamlit>=1.28.0",
    "torch>=2.5.0", 
    "transformers>=4.45.0",
    "accelerate>=0.35.0",
    "peft>=0.14.0",
    "sentencepiece>=0.2.0"
]

for package in required_packages:
    try:
        if "streamlit" in package:
            import streamlit
        elif "torch" in package:
            import torch
        elif "transformers" in package:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        elif "peft" in package:
            from peft import PeftModel, PeftConfig
    except ImportError:
        package_name = package.split(">")[0].split("=")[0]
        st.warning(f"Installing {package_name}...")
        install_package(package_name)

# Now import properly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Set page config
st.set_page_config(
    page_title="Dolly Chatbot",
    page_icon="🤖",
    layout="wide"
)

# App title and description
st.title("🤖 Dolly Chatbot")
st.markdown("Chat with your fine-tuned Dolly model!")

# Check if model files exist
if not os.path.exists("dolly_chatbot_model"):
    st.error("Model files not found! Please make sure the 'dolly_chatbot_model' folder exists with all your model files.")
    st.stop()

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model with caching"""
    try:
        with st.spinner("Loading model... This may take a minute."):
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("dolly_chatbot_model")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Try to load the model directly first
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    "dolly_chatbot_model",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            except:
                # If direct loading fails, try with PEFT
                config = PeftConfig.from_pretrained("dolly_chatbot_model")
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                model = PeftModel.from_pretrained(base_model, "dolly_chatbot_model")
            
            return tokenizer, model
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model
if not st.session_state.model_loaded:
    st.session_state.tokenizer, st.session_state.model = load_model()
    if st.session_state.tokenizer and st.session_state.model:
        st.session_state.model_loaded = True
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Failed to load model. Please check your model files.")
        st.stop()

# Function to generate response
def generate_response(instruction, input_text="", max_length=100):
    """Generate a response from the model"""
    try:
        # Format the prompt
        if input_text.strip():
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
            prompt = f"Instruction: {instruction}\nOutput:"
        
        # Tokenize input
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        if "Output:" in response:
            return response.split("Output:")[1].strip()
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Create sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    max_length = st.slider("Max response length", 50, 200, 100)
    
    st.header("ℹ️ About")
    st.info("This chatbot uses your fine-tuned Dolly model. Enter an instruction to get a response.")

# Main chat interface
st.header("💬 Chat with Dolly")

# Instruction input
instruction = st.text_area(
    "Enter your instruction:",
    placeholder="e.g., Explain machine learning in simple terms",
    height=100
)

# Optional input
input_text = st.text_area(
    "Optional input/context:",
    placeholder="e.g., For a 10-year-old audience",
    height=80
)

# Generate button
if st.button("Generate Response", type="primary"):
    if instruction.strip():
        with st.spinner("Generating response..."):
            response = generate_response(instruction, input_text, max_length)
            
            # Display response
            st.subheader("Response:")
            st.write(response)
    else:
        st.warning("Please enter an instruction first.")

# Footer
st.markdown("---")
st.caption("Powered by your fine-tuned Dolly model 🚀")
