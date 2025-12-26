"""
Product Classification Tool v3 - Streamlit Web Interface
AI-Powered Classification with Multi-Language Normalization
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.ui import get_custom_css
from src.ui.sidebar import render_sidebar
from src.ui.classification_tab import render_classification_tab
from src.ui.results_tab import render_results_tab

# Page configuration
st.set_page_config(
    page_title="Product Classification Tool v3",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize session state
if 'config' not in st.session_state:
    try:
        st.session_state.config = Config()
        st.session_state.config_loaded = True
    except Exception as e:
        st.session_state.config_loaded = False
        st.session_state.config_error = str(e)

if 'show_prompt_editor' not in st.session_state:
    st.session_state.show_prompt_editor = False

# Header
st.markdown('<div class="main-header">📦 Product Classification Tool v3</div>', unsafe_allow_html=True)
st.markdown("**AI-Powered Classification** | Multi-Language Support | Normalized Descriptions")
st.markdown("---")

# Check if config loaded successfully
if not st.session_state.config_loaded:
    st.error(f"❌ Configuration Error: {st.session_state.config_error}")
    st.info("Please check your config.ini file in the _Config folder.")
    st.stop()

config = st.session_state.config

# Render sidebar and get settings
settings = render_sidebar(config)

# Main content area - Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Classification", "📊 Results", "ℹ️ Help"])

with tab1:
    render_classification_tab(config, settings)

with tab2:
    render_results_tab()

with tab3:
    st.markdown("""
    ## 📖 How to Use
    
    ### 1. **Upload Input File**
    - Excel file (.xlsx or .xls)
    - Must have a **Description** column (mandatory)
    - Can have optional columns: Suppliers, Category, Country, etc.
    
    ### 2. **Map Columns**
    - Select which column contains product descriptions
    - Choose optional columns to include in AI analysis
    
    ### 3. **Configure Settings**
    
    #### Taxonomy (Optional)
    - Upload a taxonomy.xlsx file with predefined categories
    - If not provided, AI will suggest categories dynamically
    
    #### AI Model
    - **gpt-5-mini** (Recommended) - Best balance of cost and quality
    - **gpt-5-nano** - Cheapest option
    - **gpt-4o-mini** - Alternative option
    
    #### Web Search
    - Enable for low-confidence results
    - Choose between Google or Perplexity
    
    ### 4. **Customize Prompts (Optional)**
    - Click "Customize Prompts" to edit AI instructions
    - Test on sample rows before full run
    
    ### 5. **Run Classification**
    - Test on first N rows to verify results
    - Review cost estimate
    - Run full classification
    
    ## 📊 Output Fields
    
    ### Core Fields (Always Generated)
    - **Language** - Detected language code (es, nl, it, pt, etc.)
    - **Translated_Description** - Full English translation
    - **Compact_Noun** - Main item type (Proper Case, normalized)
    - **Normalized_Description** - Restructured format with dimensions at end
    - **Category_Type** - Functional group
    - **AI_Insight** - Brief product description
    - **AI_Confidence** - high/medium/low
    
    ### With Taxonomy
    - **AI_Category** - Exact match from taxonomy
    - **AI_Proposed_Category** - Suggested if no match
    
    ### With Web Search
    - **Web_*** - Same fields from web-enhanced classification
    
    ### Final
    - **Final_Category** - Best category determined
    - **Classification_Method** - How it was classified
    
    ## 💡 Tips
    
    1. **Start Small** - Test on 5-10 rows first
    2. **Customize Prompts** - Add domain-specific knowledge
    3. **Use Taxonomy** - For consistent categorization
    4. **Monitor Costs** - Check estimates before full runs
    5. **Review Results** - Use filters to find issues
    
    ## 🔧 Troubleshooting
    
    **Problem:** API key errors
    - Check [_Config/api_keys/](cci:7://file:///c:/AI%20Opp/Projects/_Config/api_keys:0:0-0:0) folder
    - Ensure keys are valid and have credits
    
    **Problem:** Slow classification
    - Normal for large datasets
    - Checkpoints saved every 50 rows
    
    **Problem:** Low confidence results
    - Enable web search
    - Improve input data quality
    - Customize prompts with examples
    
    **Problem:** Wrong categories
    - Review and update taxonomy
    - Add more context in optional columns
    - Customize AI prompt with domain knowledge
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Product Classification Tool v3.0 | Multi-Language | Normalization | AI + Web
    </div>
    """,
    unsafe_allow_html=True
)