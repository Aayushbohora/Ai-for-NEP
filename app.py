from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dolly Chatbot API", description="API for fine-tuned Dolly chatbot", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class ChatRequest(BaseModel):
    instruction: str
    input_text: str = ""
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

# Global variables for model and tokenizer
model = None
tokenizer = None
base_model_name = "EleutherAI/gpt-neo-125M"  # Your original base model

@app.on_event("startup")
async def load_model():
    """Load the base model and LoRA adapter when the application starts"""
    global model, tokenizer
    try:
        logger.info("Loading base model and tokenizer...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        logger.info("Loading LoRA adapter...")
        # Load the LoRA adapter
        model = PeftModel.from_pretrained(base_model, "dolly_chatbot_model")
        
        # Merge adapter with base model for faster inference
        model = model.merge_and_unload()
        
        logger.info("Model and tokenizer loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Fallback: try loading directly if merge fails
        try:
            logger.info("Trying direct loading without merge...")
            model = AutoModelForCausalLM.from_pretrained(
                "dolly_chatbot_model",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            logger.info("Model loaded directly without merge!")
        except Exception as fallback_error:
            logger.error(f"Fallback loading also failed: {str(fallback_error)}")
            raise e

@app.get("/")
async def root():
    return {"message": "Dolly Chatbot API is running!", "status": "success"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/chat", response_model=ChatResponse)
async def generate_chat_response(request: ChatRequest):
    """Generate a response based on instruction and input"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        # Format the input text
        if request.input_text.strip():
            prompt = f"### Instruction:\n{request.instruction}\n\n### Input:\n{request.input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{request.instruction}\n\n### Response:\n"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and return response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after "### Response:")
        if "### Response:" in response:
            generated_response = response.split("### Response:")[1].strip()
        else:
            generated_response = response
        
        return ChatResponse(response=generated_response)
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
