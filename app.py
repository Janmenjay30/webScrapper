import streamlit as st
import scraper_core as core

st.set_page_config(page_title="AI Web Scraper", layout="wide")

st.title("🤖 webScraper AI Agent")
st.markdown("Enter a website URL below to scrape its content and get an AI-generated summary using **Qwen 2.5 (1.5B)** running on your local GPU.")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    st.info("Model: Qwen 2.5 1.5B Instruct\nDevice: NVIDIA GTX 1650 (CUDA)")
    use_cache = st.checkbox("Cache Model", value=True)

# URL Input
url = st.text_input("Target URL", placeholder="https://example.com")

if st.button("Scrape & Summarize", type="primary"):
    if not url:
        st.error("Please enter a valid URL.")
    else:
        with st.status("Processing...", expanded=True) as status:
            # 1. Scrape
            st.write("🌐 Fetching web page...")
            text_content = core.scrape_url(url)
            
            if not text_content or len(text_content) < 100:
                status.update(label="Failed to retrieve content", state="error")
                st.error("Could not extract enough text from the page. It might be blocked or empty.")
            else:
                st.write(f"✅ Extracted {len(text_content)} characters.")
                
                # 2. Load Model
                st.write("🧠 Loading AI model...")
                # We use specific resource caching for Streamlit if needed, 
                # but our core module handles global caching too.
                # Ideally we use st.cache_resource for the model loading function.
                
                @st.cache_resource
                def get_cached_model():
                    return core.load_ai_model()
                
                model, tokenizer = get_cached_model()
                
                if not model:
                    status.update(label="Model Load Failed", state="error")
                    st.error("Could not load the AI model.")
                else:
                    # 3. Summarize
                    st.write("✨ Generating summary...")
                    input_text = text_content[:8000] # Truncate safest limit
                    summary = core.summarize_text(input_text, model, tokenizer)
                    
                    status.update(label="Complete!", state="complete")
                    
                    st.subheader("Summary Result")
                    st.success(summary)
                    
                    with st.expander("View Full Extracted Text"):
                        st.text_area("Raw Text", text_content, height=200)

st.divider()
st.caption("Powered by Hugging Face Transformers & Streamlit")
