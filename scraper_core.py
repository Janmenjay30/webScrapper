import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

# Global variables (module-level cache)
_model = None
_tokenizer = None

def load_ai_model():
    """
    Loads the AI model and tokenizer. Returns (model, tokenizer).
    """
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    print("Loading AI Model (this may take a while first time)...")
    try:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",       
            device_map="cuda"         
        )
        _model = model
        _tokenizer = tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def summarize_text(text, model, tokenizer):
    """
    Summarizes the given text using the provided model and tokenizer.
    """
    if not model or not tokenizer:
        return "Error: Model not loaded."

    # Construct the prompt using the chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web content concisely and professionally."},
        {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
    ]
    
    # Apply chat template
    text_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    # Generate
    print(f"Generating summary on {model.device}...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    # Decode only the new tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def clean_html(html_content):
    """
    Parses HTML and extracts the main readable text.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove irrelevant elements
    for script in soup(["script", "style", "nav", "footer", "header", "form", "iframe", "noscript"]):
        script.extract()

    # Strategy: Grab headers and paragraphs
    text_blocks = []
    for element in soup.find_all(['h1', 'h2', 'h3', 'p']):
        text = element.get_text(strip=True)
        if len(text) > 30: 
            text_blocks.append(text)

    return " ".join(text_blocks)

def scrape_url(url):
    """
    Fetches and cleans text from a URL. Returns the text content or None on error.
    """
    print(f"\n--- Fetching: {url} ---")
    
    # Enhanced Headers to mimic a real browser to bypass basic anti-bot checks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    
    try:
        # Use a session to maintain cookies/headers effectively
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        text_content = clean_html(response.content)
        print(f"Extracted {len(text_content)} characters of text.")
        return text_content
    except Exception as e:
        print(f"Failed to fetch page: {e}")
        return None
