import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

# 1. Setup the Analysis Agent (The "AI Model")
# We use Qwen2.5-3B-Instruct as a high-quality efficient model.
try:
    print("Loading AI Model (this may take a while first time)...")
    # Switched to 1.5B model for better stability on standard CPUs (Requires ~6GB RAM vs ~12GB for 3B)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",       # Use FP16 on GPU automatically
        device_map="cuda"         # Explicitly use CUDA/GPU
    )
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def summarize_text(text):
    # Construct the prompt using the chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web content concisely."},
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
    
    # Decode only the new tokens (removing the input prompt)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def clean_html(html_content):
    """
    Parses HTML and extracts the main readable text.
    It removes scripts, styles, navigation, and footers to focus on content.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove irrelevant elements
    for script in soup(["script", "style", "nav", "footer", "header", "form", "iframe", "noscript"]):
        script.extract()

    # Heuristic: Find the tag with the most text, likely the article body
    # Or just get all text from paragraphs if a specific container isn't obvious
    text_blocks = []
    
    # Strategy: Grab headers and paragraphs
    for element in soup.find_all(['h1', 'h2', 'h3', 'p']):
        text = element.get_text(strip=True)
        if len(text) > 30: # Filter out short snippets/menu items
            text_blocks.append(text)

    return " ".join(text_blocks)

def scrape_and_analyze(url):
    print(f"\n--- Fetching: {url} ---")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch page: {e}")
        return

    # Extract Text
    text_content = clean_html(response.content)
    print(f"Extracted {len(text_content)} characters of text.")
    
    if len(text_content) < 100:
        print("Not enough text found to analyze. Ensure the site isn't pure JavaScript.")
        return

    # Truncate for the model (most small models handle ~1024 tokens, approx 3-4k chars)
    # We take the beginning of the article as it usually contains the main point.
    input_text = text_content[:8000] # Increased for Qwen


    print("\n--- Running AI Analysis (Summarization) ---")
    try:
        # Generate summary
        summary_text = summarize_text(input_text)
        
        print("\n=== AI Summary ===")
        print(summary_text)
        print("==================\n")
        
        # Save to file
        with open("ai_analysis_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Source: {url}\n")
            f.write(f"Summary: {summary_text}\n")
            f.write("-" * 50 + "\n")
            
    except Exception as e:
        print(f"AI Model Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
    else:
        target_url = input("Enter a URL to scrape and analyze: ")
    
    scrape_and_analyze(target_url)
