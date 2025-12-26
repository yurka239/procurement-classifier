"""
Sidebar UI Components
"""

import streamlit as st
from pathlib import Path

def render_sidebar(config):
    """Render the sidebar with settings"""
    
    st.sidebar.title("⚙️ Settings")
    
    # File Upload Section
    st.sidebar.markdown("### 📁 Input File")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel file",
        type=['xlsx', 'xls'],
        help="File must contain a 'Description' column"
    )
    
    # Column Mapping
    if uploaded_file:
        import pandas as pd
        df = pd.read_excel(uploaded_file)
        
        st.sidebar.markdown("### 🗂️ Column Mapping")
        
        # Description column (mandatory)
        desc_col = st.sidebar.selectbox(
            "Description Column *",
            options=df.columns.tolist(),
            index=df.columns.tolist().index('Description') if 'Description' in df.columns else 0
        )
        
        # Optional columns
        other_cols = [col for col in df.columns if col != desc_col]
        
        st.sidebar.markdown("**Optional Columns:**")
        selected_cols = st.sidebar.multiselect(
            "Include these columns in AI prompt",
            options=other_cols,
            default=[col for col in ['Suppliers', 'Category', 'Orig_Category'] if col in other_cols]
        )
        
        # Store in session state
        st.session_state.column_mapping = {
            'description': desc_col,
            'optional': selected_cols
        }
        st.session_state.input_df = df
    
    st.sidebar.markdown("---")
    
    # Taxonomy Section
    st.sidebar.markdown("### 🎯 Taxonomy")
    use_taxonomy = st.sidebar.checkbox(
        "Use predefined categories",
        value=False,
        help="Upload a taxonomy file with predefined categories"
    )
    
    taxonomy_file = None
    if use_taxonomy:
        taxonomy_file = st.sidebar.file_uploader(
            "Upload taxonomy.xlsx",
            type=['xlsx'],
            help="File must have a 'Category' column"
        )
        
        if taxonomy_file:
            st.session_state.taxonomy_file = taxonomy_file
    
    st.sidebar.markdown("---")
    
    # AI Model Settings
    st.sidebar.markdown("### 🤖 AI Model")
    
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=config.available_models,
        index=config.available_models.index(config.default_model) if config.default_model in config.available_models else 0,
        help="GPT-5-mini is recommended for classification"
    )
    
    # Show pricing
    if model_name in config.price_table:
        prices = config.price_table[model_name]
        st.sidebar.caption(f"💰 ${prices['input']}/M input, ${prices['output']}/M output")
    
    # Confidence threshold
    st.sidebar.markdown("### 🎚️ Confidence")
    min_confidence = st.sidebar.select_slider(
        "Trigger web search at",
        options=['high', 'medium', 'low', 'never'],
        value=config.min_confidence_for_web,
        help="When to use web search as fallback"
    )
    
    st.sidebar.markdown("---")
    
    # Web Search Settings
    st.sidebar.markdown("### 🔍 Web Search")
    
    use_web = st.sidebar.checkbox(
        "Enable web search",
        value=config.use_web_search,
        help="Use web search for low confidence results"
    )
    
    if use_web:
        web_provider = st.sidebar.radio(
            "Provider",
            options=['google', 'perplexity'],
            index=0,  # Google is default (first option)
            help="Google Custom Search or Perplexity AI"
        )
        st.session_state.web_provider = web_provider
    
    st.sidebar.markdown("---")
    
    # Prompt Customization
    st.sidebar.markdown("### ✏️ Prompts")
    
    if st.sidebar.button("📝 Customize Prompts"):
        st.session_state.show_prompt_editor = True
    
    # Store settings in session state
    st.session_state.settings = {
        'model_name': model_name,
        'use_taxonomy': use_taxonomy,
        'use_web_search': use_web,
        'min_confidence': min_confidence,
        'uploaded_file': uploaded_file,
        'taxonomy_file': taxonomy_file if use_taxonomy else None
    }
    
    return st.session_state.settings