"""Procurement Classifier v2.1 - Enhanced Streamlit UI

Features:
- Structured attribute extraction and filtering
- Normalization visualization (before/after)
- Benchmarking-ready normalized descriptions
- Supplier comparison by attributes
- Export with customizable columns
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import io
import configparser

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page config
st.set_page_config(
    page_title="Procurement Classifier v2.1",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.2rem; color: #666; margin-bottom: 2rem;}
    .setup-box {background: #f0f8ff; padding: 2rem; border-radius: 10px; border: 2px solid #1f77b4;}
</style>
""", unsafe_allow_html=True)

def check_config_needs_setup():
    """Check if config.ini exists and has valid API key"""
    config_path = Path(__file__).parent / '_Config' / 'config.ini'
    example_path = Path(__file__).parent / '_Config' / 'config.ini.example'
    
    # If no config.ini, needs setup
    if not config_path.exists():
        # Create from example if available
        if example_path.exists():
            import shutil
            shutil.copy(example_path, config_path)
        return True, config_path
    
    # Check if API key is placeholder
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    openai_key = config.get('API_Keys', 'openai_api_key', fallback='').strip()
    
    # Check for placeholder or empty
    if not openai_key or 'your-' in openai_key.lower() or not openai_key.startswith('sk-'):
        return True, config_path
    
    return False, config_path

def get_env_key(env_var):
    """Get API key from environment variable"""
    import os
    return os.environ.get(env_var, '').strip()

def read_key_from_file(file_path):
    """Read API key from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except:
        return ''

def show_setup_wizard(config_path):
    """Show first-time setup wizard with multiple configuration options"""
    import os
    
    st.markdown('<div class="main-header">🏭 Procurement Classifier v2.1</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">First-Time Setup</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.info("👋 **Welcome!** Let's set up your API keys to get started.")
    
    # Check for environment variables
    env_openai = get_env_key('OPENAI_API_KEY')
    env_google = get_env_key('GOOGLE_API_KEY')
    env_google_cse = get_env_key('GOOGLE_CSE_ID')
    
    with st.container():
        st.subheader("🔑 API Key Configuration")
        
        st.markdown("""
        **Required:** OpenAI API Key  
        **Optional:** Google API Keys (for web search enhancement)
        """)
        
        st.markdown("---")
        
        # ========== OpenAI Key ==========
        st.markdown("### 1. OpenAI API Key (Required)")
        st.markdown("Get your key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)")
        
        openai_method = st.radio(
            "How would you like to provide your OpenAI API key?",
            ["📁 Upload key file", "📝 Paste key directly", "🌐 Use environment variable"],
            key="openai_method",
            horizontal=True
        )
        
        openai_key = ""
        
        if openai_method == "📁 Upload key file":
            openai_file = st.file_uploader(
                "Select your OpenAI API key file (.txt)",
                type=['txt'],
                key="openai_file_upload",
                help="Upload a text file containing your OpenAI API key"
            )
            if openai_file:
                openai_key = openai_file.read().decode('utf-8').strip()
                if openai_key:
                    st.success(f"✅ Key loaded: {openai_key[:15]}...")
                else:
                    st.warning("⚠️ File is empty")
        
        elif openai_method == "📝 Paste key directly":
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-proj-...",
                help="Your OpenAI API key starting with 'sk-'"
            )
        
        else:  # Environment variable
            if env_openai:
                st.success(f"✅ Found OPENAI_API_KEY: {env_openai[:15]}...")
                openai_key = env_openai
            else:
                st.warning("⚠️ OPENAI_API_KEY environment variable not set")
        
        st.markdown("---")
        
        # ========== Google Keys (Optional) ==========
        st.markdown("### 2. Google Search API (Optional)")
        
        with st.expander("Configure Google API for web search enhancement", expanded=False):
            st.markdown("Get your keys from [Google Cloud Console](https://console.cloud.google.com/)")
            st.markdown("You need **2 items**: API Key + Custom Search Engine ID")
            
            google_method = st.radio(
                "How would you like to provide Google API keys?",
                ["⏭️ Skip (not using web search)", "📁 Upload key files", "📝 Paste keys directly", "🌐 Use environment variables"],
                key="google_method"
            )
            
            google_key = ""
            google_cse = ""
            
            if google_method == "📁 Upload key files":
                col1, col2 = st.columns(2)
                with col1:
                    google_key_file = st.file_uploader(
                        "Google API Key file (.txt)",
                        type=['txt'],
                        key="google_key_upload"
                    )
                    if google_key_file:
                        google_key = google_key_file.read().decode('utf-8').strip()
                        if google_key:
                            st.success(f"✅ Key loaded: {google_key[:10]}...")
                
                with col2:
                    google_cse_file = st.file_uploader(
                        "Google CSE ID file (.txt)",
                        type=['txt'],
                        key="google_cse_upload"
                    )
                    if google_cse_file:
                        google_cse = google_cse_file.read().decode('utf-8').strip()
                        if google_cse:
                            st.success(f"✅ CSE ID loaded: {google_cse[:10]}...")
            
            elif google_method == "📝 Paste keys directly":
                col1, col2 = st.columns(2)
                with col1:
                    google_key = st.text_input(
                        "Google API Key",
                        type="password",
                        placeholder="AIza..."
                    )
                with col2:
                    google_cse = st.text_input(
                        "Google CSE ID",
                        placeholder="abc123..."
                    )
            
            elif google_method == "🌐 Use environment variables":
                if env_google:
                    st.success(f"✅ GOOGLE_API_KEY found")
                    google_key = env_google
                else:
                    st.warning("⚠️ GOOGLE_API_KEY not set")
                
                if env_google_cse:
                    st.success(f"✅ GOOGLE_CSE_ID found")
                    google_cse = env_google_cse
                else:
                    st.warning("⚠️ GOOGLE_CSE_ID not set")
        
        st.markdown("---")
        
        # ========== Save Button ==========
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 Save Configuration & Start", type="primary", use_container_width=True):
                # Validate OpenAI key (openai_key is set from any of the 3 methods)
                if not openai_key or not openai_key.startswith('sk-'):
                    st.error("⚠️ Valid OpenAI API Key is required! (must start with 'sk-')")
                    return
                
                # Load existing config and update
                config = configparser.ConfigParser()
                config.read(config_path, encoding='utf-8')
                
                # Ensure sections exist
                if 'API_Keys' not in config:
                    config.add_section('API_Keys')
                if 'Paths' not in config:
                    config.add_section('Paths')
                
                # Save OpenAI key directly (from file upload, paste, or env var)
                config.set('API_Keys', 'openai_api_key', openai_key)
                
                # Save Google keys if provided
                if google_key:
                    config.set('API_Keys', 'google_api_key', google_key)
                if google_cse:
                    config.set('API_Keys', 'google_cse_id', google_cse)
                
                # Save config
                with open(config_path, 'w', encoding='utf-8') as f:
                    config.write(f)
                
                st.success("✅ Configuration saved!")
                st.balloons()
                st.info("🔄 Reloading app...")
                st.rerun()
    
    st.stop()

# Check if setup needed
needs_setup, config_path = check_config_needs_setup()

if needs_setup:
    show_setup_wizard(config_path)

# Normal app flow - import config after setup check
from src.config import Config
from src.classifier_engine import ClassificationEngine
from src.attribute_config import ATTRIBUTE_CATEGORIES, PRESET_SELECTIONS, get_all_attributes, build_prompt_section

# Initialize session state
if 'classified_df' not in st.session_state:
    st.session_state.classified_df = None
if 'project_dir' not in st.session_state:
    st.session_state.project_dir = None
if 'stats' not in st.session_state:
    st.session_state.stats = None

# Header
st.markdown('<div class="main-header">🏭 Procurement Classifier v2.1</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Classification with Attribute Extraction & Normalization</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    try:
        config = Config()
        st.success("✅ Config loaded")
    except Exception as e:
        st.error(f"❌ Config error: {e}")
        st.stop()
    
    if not config.openai_key:
        st.error("⚠️ OpenAI API key not found!")
        st.info("Add your key to `_Config/config.ini`")
        st.stop()
    else:
        st.success("🔑 API key configured")
    
    st.divider()
    
    st.subheader("🤖 Model Settings")
    model_options = config.available_models  # Already a list from config.py
    selected_model = st.selectbox(
        "AI Model",
        options=model_options,
        index=model_options.index(config.default_model) if config.default_model in model_options else 0
    )
    
    st.subheader("🎯 Features")
    use_web_search = st.checkbox("Web Search Enhancement", value=config.use_web_search)
    extract_attributes = st.checkbox("Extract Attributes", value=getattr(config, 'extract_attributes', True))
    enable_normalization = st.checkbox("Enable Normalization", value=getattr(config, 'enable_normalization', True))
    
    if enable_normalization:
        fingerprint_threshold = st.slider("Clustering Threshold", 0.75, 0.95, 
                                         getattr(config, 'fingerprint_threshold', 0.85), 0.05)
    
    st.divider()
    
    # Hierarchical attribute selection with categories, groups, and custom attributes
    if extract_attributes:
        from src.attribute_config import get_attribute_info, get_category_attributes
        
        # Default MRO-based attributes
        DEFAULT_ATTRS = ['Material', 'Material_Grade', 'Size_Dimension', 'Part_Model_Number', 
                        'UOM', 'Quantity_Per_Pack', 'Pressure_Rating', 'Thread_Type']
        
        # Get all available attributes
        all_attrs = get_all_attributes()
        
        # Initialize session state
        if 'selected_attributes' not in st.session_state:
            st.session_state['selected_attributes'] = DEFAULT_ATTRS.copy()
        if 'custom_attributes' not in st.session_state:
            st.session_state['custom_attributes'] = {}
        
        st.markdown("**📋 Attributes to Extract**")
        
        # Select All / Clear All / Reset buttons
        ctrl_cols = st.columns(3)
        with ctrl_cols[0]:
            if st.button("✅ Select All", key="btn_select_all", use_container_width=True):
                st.session_state['selected_attributes'] = all_attrs.copy()
                st.rerun()
        with ctrl_cols[1]:
            if st.button("❌ Clear All", key="btn_clear_all", use_container_width=True):
                st.session_state['selected_attributes'] = []
                st.rerun()
        with ctrl_cols[2]:
            if st.button("🔄 Reset Default", key="btn_reset", use_container_width=True):
                st.session_state['selected_attributes'] = DEFAULT_ATTRS.copy()
                st.rerun()
        
        # Current selection for reference
        current_selection = set(st.session_state.get('selected_attributes', DEFAULT_ATTRS))
        
        # Hierarchical category expanders
        for cat_key, cat_info in ATTRIBUTE_CATEGORIES.items():
            cat_attrs = list(cat_info['attributes'].keys())
            selected_in_cat = [a for a in cat_attrs if a in current_selection]
            all_selected = len(selected_in_cat) == len(cat_attrs)
            none_selected = len(selected_in_cat) == 0
            
            # Category header with selection indicator
            cat_label = f"{cat_info['icon']} {cat_info['display_name']} ({len(selected_in_cat)}/{len(cat_attrs)})"
            
            with st.expander(cat_label, expanded=len(selected_in_cat) > 0):
                # Group-level checkbox
                group_cols = st.columns([3, 1, 1])
                with group_cols[1]:
                    if st.button("All", key=f"sel_all_{cat_key}", use_container_width=True):
                        for attr in cat_attrs:
                            if attr not in st.session_state['selected_attributes']:
                                st.session_state['selected_attributes'].append(attr)
                        st.rerun()
                with group_cols[2]:
                    if st.button("None", key=f"sel_none_{cat_key}", use_container_width=True):
                        st.session_state['selected_attributes'] = [
                            a for a in st.session_state['selected_attributes'] if a not in cat_attrs
                        ]
                        st.rerun()
                
                # Individual attribute checkboxes with descriptions
                for attr_name, attr_info in cat_info['attributes'].items():
                    is_selected = attr_name in current_selection
                    help_text = f"{attr_info['description']} (e.g., {attr_info['examples']})"
                    
                    if st.checkbox(attr_name.replace('_', ' '), value=is_selected, 
                                  key=f"attr_{cat_key}_{attr_name}", help=help_text):
                        if attr_name not in st.session_state['selected_attributes']:
                            st.session_state['selected_attributes'].append(attr_name)
                    else:
                        if attr_name in st.session_state['selected_attributes']:
                            st.session_state['selected_attributes'].remove(attr_name)
        
        # Custom Attributes Section
        with st.expander("➕ Custom Attributes", expanded=len(st.session_state['custom_attributes']) > 0):
            st.caption("Create custom attributes for specialized extraction needs")
            
            # Display existing custom attributes
            if st.session_state['custom_attributes']:
                st.markdown("**Your Custom Attributes:**")
                for custom_name, custom_info in list(st.session_state['custom_attributes'].items()):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.write(f"• **{custom_name}**: {custom_info['description']}")
                        if custom_info.get('examples'):
                            st.caption(f"  Examples: {custom_info['examples']}")
                    with c2:
                        if st.button("🗑️", key=f"del_custom_{custom_name}"):
                            del st.session_state['custom_attributes'][custom_name]
                            if custom_name in st.session_state['selected_attributes']:
                                st.session_state['selected_attributes'].remove(custom_name)
                            st.rerun()
                st.divider()
            
            # Add new custom attribute form
            st.markdown("**Add New Custom Attribute:**")
            new_attr_name = st.text_input("Attribute Name", key="new_custom_attr_name",
                                         placeholder="e.g., Certification_Type")
            new_attr_desc = st.text_input("Description (for AI understanding)", key="new_custom_attr_desc",
                                         placeholder="e.g., Type of product certification or compliance standard")
            new_attr_examples = st.text_input("Examples (comma-separated)", key="new_custom_attr_examples",
                                             placeholder="e.g., ISO 9001, FDA approved, CE marked")
            
            if st.button("➕ Add Custom Attribute", key="btn_add_custom"):
                if new_attr_name and new_attr_desc:
                    # Sanitize name (replace spaces with underscores)
                    clean_name = new_attr_name.strip().replace(' ', '_')
                    if clean_name not in all_attrs and clean_name not in st.session_state['custom_attributes']:
                        st.session_state['custom_attributes'][clean_name] = {
                            'key': clean_name.lower(),
                            'description': new_attr_desc.strip(),
                            'examples': new_attr_examples.strip()
                        }
                        st.session_state['selected_attributes'].append(clean_name)
                        st.success(f"✅ Added: {clean_name}")
                        st.rerun()
                    else:
                        st.error("❌ Attribute already exists!")
                else:
                    st.warning("⚠️ Name and description are required")
        
        # Summary
        total_selected = len(st.session_state['selected_attributes'])
        total_available = len(all_attrs) + len(st.session_state['custom_attributes'])
        custom_count = len(st.session_state['custom_attributes'])
        
        if total_selected > 0:
            savings = ((len(all_attrs) - total_selected) / len(all_attrs)) * 100 if total_selected <= len(all_attrs) else 0
            summary = f"✅ {total_selected} selected"
            if custom_count > 0:
                summary += f" ({custom_count} custom)"
            if savings > 0:
                summary += f" | 💰 ~{savings:.0f}% token savings"
            st.caption(summary)
        else:
            st.warning("⚠️ No attributes selected!")
    else:
        st.session_state['selected_attributes'] = []
        st.session_state['custom_attributes'] = {}
    
    st.divider()
    
    with st.expander("⚡ Performance"):
        max_workers = st.number_input("Parallel Workers", 1, 20, getattr(config, 'max_workers', 10))
        checkpoint_every = st.number_input("Checkpoint Every N Rows", 0, 500, 
                                          getattr(config, 'save_checkpoint_every', 100))

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Classify", "📊 Results", "🔍 Attributes", "📈 Benchmarking"])

# TAB 1: Upload & Classify
with tab1:
    st.header("Upload Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Input File")
        input_file = st.file_uploader("Upload Excel file (.xlsx)", type=['xlsx'])
        
        if input_file:
            st.success(f"✅ Loaded: {input_file.name}")
            try:
                preview_df = pd.read_excel(input_file, nrows=5)
                st.write("**Preview:**")
                st.dataframe(preview_df, use_container_width=True)
                if 'Description' not in preview_df.columns:
                    st.error("❌ Missing 'Description' column!")
                else:
                    # Show additional columns configuration
                    st.divider()
                    st.subheader("🔧 Additional Context Columns")
                    st.info("Select columns to help AI understand the context better. You can rename them to make their purpose clearer.")
                    
                    # Get available columns (exclude Index and Description)
                    available_cols = [col for col in preview_df.columns if col not in ['Index', 'Description']]
                    
                    if available_cols:
                        # Store column mappings in session state
                        if 'column_mappings' not in st.session_state:
                            st.session_state.column_mappings = {}
                        
                        # Multi-select for columns to include
                        selected_cols = st.multiselect(
                            "Select columns to include as context",
                            available_cols,
                            default=[col for col in available_cols if col in ['Supplier', 'Material Group', 'Country', 'Category']],
                            help="These columns will be passed to AI to improve classification"
                        )
                        
                        if selected_cols:
                            st.write("**Rename columns for AI (optional):**")
                            st.caption("Give columns descriptive names to help AI understand their purpose (e.g., 'Mat_Group' → 'Material Group', 'Cntry' → 'Country')")
                            
                            # Create rename inputs
                            col_rename = {}
                            cols_per_row = 2
                            for i in range(0, len(selected_cols), cols_per_row):
                                cols = st.columns(cols_per_row)
                                for j, col_name in enumerate(selected_cols[i:i+cols_per_row]):
                                    with cols[j]:
                                        new_name = st.text_input(
                                            f"Rename '{col_name}'",
                                            value=st.session_state.column_mappings.get(col_name, col_name),
                                            key=f"rename_{col_name}",
                                            placeholder=col_name
                                        )
                                        col_rename[col_name] = new_name if new_name else col_name
                            
                            # Store mappings
                            st.session_state.column_mappings = col_rename
                            st.session_state.selected_columns = selected_cols
                            
                            # Show preview of what will be sent to AI
                            with st.expander("📋 Preview: What AI will see"):
                                example_row = preview_df.iloc[0]
                                context_preview = "Description: " + str(example_row.get('Description', ''))
                                context_preview += "\n\nADDITIONAL CONTEXT:\n"
                                for orig_col in selected_cols:
                                    renamed = col_rename[orig_col]
                                    value = example_row.get(orig_col, '')
                                    if pd.notna(value) and str(value).strip():
                                        context_preview += f"{renamed}: {value}\n"
                                st.code(context_preview, language="text")
                    else:
                        st.warning("No additional columns found in the file.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.subheader("📋 Taxonomy (Optional)")
        taxonomy_file = st.file_uploader("Upload taxonomy (.xlsx)", type=['xlsx'])
        if taxonomy_file:
            st.success(f"✅ Loaded: {taxonomy_file.name}")
    
    st.divider()
    project_name = st.text_input("Project Name (optional)", placeholder="e.g., Q4_2024_Procurement")
    
    if st.button("🚀 Start Classification", type="primary", use_container_width=True):
        if not input_file:
            st.error("Please upload an input file!")
        else:
            input_path = Path("temp_input.xlsx")
            with open(input_path, "wb") as f:
                f.write(input_file.getvalue())
            
            taxonomy_path = None
            if taxonomy_file:
                taxonomy_path = Path("temp_taxonomy.xlsx")
                with open(taxonomy_path, "wb") as f:
                    f.write(taxonomy_file.getvalue())
            
            config.extract_attributes = extract_attributes
            config.enable_normalization = enable_normalization
            if enable_normalization:
                config.fingerprint_threshold = fingerprint_threshold
            config.max_workers = max_workers
            config.save_checkpoint_every = checkpoint_every
            
            # Pass column selection and mappings if configured
            if 'selected_columns' in st.session_state:
                config.selected_columns = st.session_state.selected_columns
            if 'column_mappings' in st.session_state:
                config.column_mappings = st.session_state.column_mappings
            
            engine = ClassificationEngine(config, model_name=selected_model)
            status_text = st.empty()
            
            def progress_callback(msg):
                status_text.write(msg)
            
            try:
                with st.spinner("🔄 Processing..."):
                    # Get selected attributes and custom attributes from session state
                    selected_attrs = st.session_state.get('selected_attributes', None)
                    custom_attrs = st.session_state.get('custom_attributes', {})
                    
                    output_file, project_dir, stats = engine.classify_batch(
                        input_file=input_path,
                        taxonomy_file=taxonomy_path,
                        project_name=project_name if project_name else None,
                        progress_callback=progress_callback,
                        use_web_search=use_web_search,
                        save_to_disk=True,
                        attributes_to_extract=selected_attrs if selected_attrs else None,
                        custom_attributes=custom_attrs if custom_attrs else None
                    )
                
                st.session_state.classified_df = pd.read_excel(output_file)
                st.session_state.project_dir = project_dir
                st.session_state.stats = stats
                
                st.balloons()
                st.success("✅ Classification Complete!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", stats['total_rows'])
                with col2:
                    st.metric("Time (s)", f"{stats['elapsed_time']:.1f}")
                with col3:
                    st.metric("Cost ($)", f"${stats['cost']:.4f}")
                with col4:
                    st.metric("Failed", stats.get('failed_rows', 0))
                
                if enable_normalization:
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Unique Nouns", stats.get('unique_nouns', 0))
                    with col2:
                        st.metric("Unique Categories", stats.get('unique_categories', 0))
                    with col3:
                        st.metric("Unique Brands", stats.get('unique_brands', 0))
                
                st.info(f"📁 Results: `{project_dir}`")
                
                input_path.unlink(missing_ok=True)
                if taxonomy_path:
                    taxonomy_path.unlink(missing_ok=True)
                
            except Exception as e:
                st.error(f"❌ Failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# TAB 2: Results
with tab2:
    st.header("Classification Results")
    
    if st.session_state.classified_df is None:
        st.info("👈 Classify data first to see results.")
    else:
        df = st.session_state.classified_df
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", len(df))
        with col2:
            high_conf = len(df[df['AI_Confidence'] == 'high'])
            st.metric("High Confidence", high_conf)
        with col3:
            products = len(df[df['Is_Product_or_Service'] == 'Product'])
            st.metric("Products", products)
        with col4:
            services = len(df[df['Is_Product_or_Service'] == 'Service'])
            st.metric("Services", services)
        
        st.divider()
        st.subheader("🔍 Filters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            conf_filter = st.multiselect("Confidence", ['high', 'medium', 'low'], 
                                        default=['high', 'medium', 'low'])
        with col2:
            type_filter = st.multiselect("Type", ['Product', 'Service'], 
                                        default=['Product', 'Service'])
        with col3:
            if 'Needs_Review' in df.columns:
                review_filter = st.selectbox("Review", ['All', 'Needs Review', 'OK'])
        
        filtered_df = df[(df['AI_Confidence'].isin(conf_filter)) & 
                        (df['Is_Product_or_Service'].isin(type_filter))]
        
        if 'Needs_Review' in df.columns and review_filter != 'All':
            filtered_df = filtered_df[filtered_df['Needs_Review'] == (1 if review_filter == 'Needs Review' else 0)]
        
        st.info(f"Showing {len(filtered_df)} of {len(df)} rows")
        
        default_cols = ['Index', 'Description', 'Concept_Noun', 'Category_Type', 'Brand', 
                       'Normalized_Description', 'AI_Confidence']
        available = [c for c in default_cols if c in df.columns]
        
        selected_cols = st.multiselect("Display Columns", df.columns.tolist(), default=available)
        
        st.dataframe(filtered_df[selected_cols] if selected_cols else filtered_df, 
                    use_container_width=True, height=500)
        
        # Download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False)
        
        st.download_button("📥 Download Results", buffer.getvalue(), 
                          "filtered_results.xlsx", 
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# TAB 3: Attributes
with tab3:
    st.header("Attribute Analysis")
    
    if st.session_state.classified_df is None:
        st.info("👈 Classify data first.")
    else:
        df = st.session_state.classified_df
        
        # Get all possible attribute columns from dynamic config + custom attributes
        all_possible_attrs = get_all_attributes()
        custom_attr_names = list(st.session_state.get('custom_attributes', {}).keys())
        all_possible_attrs.extend(custom_attr_names)
        
        # Find which attributes exist in the dataframe
        attr_cols = [c for c in df.columns if c in all_possible_attrs]
        
        if not attr_cols:
            st.warning("⚠️ No attributes found. Enable extraction in settings.")
        else:
            st.info(f"📊 Found {len(attr_cols)} extracted attributes in results")
            
            selected_attr = st.selectbox("Select Attribute to Analyze", attr_cols)
            
            # Filter out empty/NaN values for analysis
            value_counts = df[selected_attr].dropna().replace('', pd.NA).dropna().value_counts().head(20)
            
            if len(value_counts) == 0:
                st.info(f"No values found for {selected_attr}")
            else:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart(value_counts)
                with col2:
                    st.write("**Top Values:**")
                    for val, count in value_counts.items():
                        if val:
                            st.write(f"• {val}: {count}")

# TAB 4: Benchmarking
with tab4:
    st.header("Supplier Benchmarking")
    
    if st.session_state.classified_df is None:
        st.info("👈 Classify data first.")
    else:
        df = st.session_state.classified_df
        
        if 'Normalized_Description' not in df.columns:
            st.warning("⚠️ Enable normalization to see benchmarking.")
        else:
            grouped = df.groupby('Normalized_Description').agg({
                'Description': 'count',
                'Brand': lambda x: ', '.join(x.dropna().unique()[:3]),
                'Concept_Noun': 'first'
            }).reset_index()
            
            grouped.columns = ['Normalized_Description', 'Count', 'Brands', 'Concept_Noun']
            grouped = grouped.sort_values('Count', ascending=False)
            
            st.write("**Most Common Items:**")
            st.dataframe(grouped.head(20), use_container_width=True)
            
            st.divider()
            selected_item = st.selectbox("Drill-down", grouped['Normalized_Description'].tolist())
            
            if selected_item:
                item_df = df[df['Normalized_Description'] == selected_item]
                st.write(f"**Found {len(item_df)} variations:**")
                st.dataframe(item_df[['Description', 'Brand', 'Concept_Noun', 'Modifier']], 
                           use_container_width=True)
