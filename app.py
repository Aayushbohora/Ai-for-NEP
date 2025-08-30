import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import re

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

# Initialize session state for model loading
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model with caching to avoid reloading on every interaction"""
    try:
        with st.spinner("Loading Nexo AI model... This may take a minute."):
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("dolly_chatbot_model")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load the base model config
            config = PeftConfig.from_pretrained("dolly_chatbot_model")
            base_model_name = config.base_model_name_or_path
            
            # Load base model with memory optimization
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, "dolly_chatbot_model")
            
            return tokenizer, model
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model only once
if not st.session_state.model_loaded:
    st.session_state.tokenizer, st.session_state.model = load_model()
    if st.session_state.tokenizer and st.session_state.model:
        st.session_state.model_loaded = True
        st.success("✅ Nexo AI model loaded successfully!")
    else:
        st.error("❌ Failed to load model. Please check your model files.")

def clean_response(response):
    """Clean up the model response to remove unwanted patterns"""
    # Remove any incomplete sentences at the end
    response = re.sub(r'[^.!?]+$', '', response)
    
    # Remove any markdown formatting if present
    response = re.sub(r'\*\*|\*|__|_', '', response)
    
    # Remove any numbered lists that don't make sense
    if response.startswith('1.') and len(response.split('\n')) < 3:
        # If it's a very short list, it's probably not a good response
        lines = response.split('\n')
        if len(lines) > 0:
            response = lines[0].replace('1. ', '', 1)
    
    # Trim whitespace
    response = response.strip()
    
    return response

def generate_response(user_input, max_length=150, temperature=0.7):
    """Generate a response from the model"""
    if not st.session_state.model_loaded:
        return "Model not loaded yet. Please wait..."
    
    try:
        # Improved prompt engineering
        if user_input.lower().startswith(('explain', 'what is', 'what are', 'how', 'why')):
            prompt = f"Please provide a clear and concise explanation for: {user_input}\n\nResponse:"
        elif user_input.lower().startswith(('write', 'create', 'generate')):
            prompt = f"Please create the following: {user_input}\n\nResponse:"
        elif user_input.lower().startswith(('list', 'name some', 'give examples')):
            prompt = f"Please provide a list for: {user_input}\n\nResponse:"
        else:
            prompt = f"Please respond to the following: {user_input}\n\nResponse:"
        
        # Tokenize input
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response with better parameters
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
        
        # Decode response
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after "Response:")
        if "Response:" in response:
            response = response.split("Response:")[1].strip()
        
        # Clean up the response
        response = clean_response(response)
        
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Create sidebar for settings and info
with st.sidebar:
    st.header("⚙️ Settings")
    max_length = st.slider("Response Length", 50, 300, 150)
    temperature = st.slider("Creativity", 0.1, 1.0, 0.7)
    
    st.markdown("---")
    st.header("ℹ️ About Nexo")
    st.info("""
    **Nexo** is an AI assistant created by **Aayush Bohora**. 
    It's powered by a fine-tuned language model specialized in helpful and accurate responses.
    """)
    
    st.markdown("---")
    st.caption("Version 1.0 • Built with Streamlit")

# Main chat interface
st.header("💬 Chat with Nexo")

# Single input text field
user_input = st.text_area(
    "Enter your question or instruction:",
    placeholder="e.g., Explain machine learning in simple terms\nor\nWrite a poem about artificial intelligence\nor\nHow do I make chocolate chip cookies?",
    height=120,
    key="user_input"
)

# Generate button
if st.button("Get Response", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("Nexo is thinking..."):
            response = generate_response(user_input, max_length, temperature)
            
            # Display response in a nice container
            st.subheader("Nexo's Response:")
            
            # Check if the response is poor quality
            if any(word in response.lower() for word in ['wedding', 'party', 'band', 'venue']) and not any(word in user_input.lower() for word in ['wedding', 'party', 'celebration']):
                st.warning("⚠️ The response doesn't seem relevant to your question. This might be due to model limitations.")
                st.info("💡 Try rephrasing your question or asking something different.")
            
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
