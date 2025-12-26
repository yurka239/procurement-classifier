"""
Main Classification Tab UI
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
from src.classifier_engine import ClassificationEngine
from src.ui.utils import get_default_ai_prompt, get_default_web_prompt, estimate_cost

def render_classification_tab(config, settings):
    """Render the main classification interface"""
    
    st.header("📊 Product Classification")
    
    # Check if file is uploaded
    if not settings.get('uploaded_file'):
        st.info("👈 Please upload an input file in the sidebar to begin")
        return
    
    # Show file preview
    if 'input_df' in st.session_state:
        df = st.session_state.input_df
        
        st.markdown("### 📋 Input Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 of {len(df)} rows")
        
        # Show column mapping
        if 'column_mapping' in st.session_state:
            mapping = st.session_state.column_mapping
            st.markdown("**Column Mapping:**")
            cols = st.columns(2)
            with cols[0]:
                st.info(f"📝 Description: `{mapping['description']}`")
            with cols[1]:
                if mapping['optional']:
                    st.info(f"📎 Optional: {', '.join([f'`{c}`' for c in mapping['optional']])}")
        
        st.markdown("---")
        
        # Prompt Editor
        if st.session_state.get('show_prompt_editor', False):
            render_prompt_editor()
            st.markdown("---")
        
        # Test on sample
        st.markdown("### 🧪 Test Classification")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            test_rows = st.number_input("Test on first N rows", min_value=1, max_value=min(20, len(df)), value=5)
        with col2:
            if st.button("▶️ Run Test", type="secondary"):
                # Store test request in session state
                st.session_state.run_test = True
                st.session_state.test_rows = test_rows
        
        # Run test outside column layout (full width)
        if st.session_state.get('run_test', False):
            run_test_classification(config, settings, st.session_state.test_rows)
            st.session_state.run_test = False  # Reset flag
        
        st.markdown("---")
        
        # Cost Estimate
        st.markdown("### 💰 Cost & Time Estimate")
        estimate = estimate_cost(len(df), settings['model_name'], config.price_table)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Model", settings['model_name'])
        with col3:
            st.metric("Estimated Cost", estimate['cost_formatted'])
        with col4:
            st.metric("Estimated Time", estimate['time_formatted'])
        
        st.caption("⚠️ Actual cost may vary based on product complexity and web search usage")
        
        st.markdown("---")
        
        # Start Classification
        st.markdown("### 🚀 Start Classification")
        
        # Initialize default project name in session state
        if 'project_name' not in st.session_state:
            st.session_state.project_name = f"Classification_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        project_name = st.text_input(
            "Project Name (optional)",
            value=st.session_state.project_name,
            key="project_name_input",
            help="Leave empty for auto-generated name"
        )
        
        # Update session state when user types
        st.session_state.project_name = project_name
        
        if st.button("▶️ START CLASSIFICATION", type="primary", use_container_width=True):
            run_full_classification(config, settings, project_name)


def render_prompt_editor():
    """Render the prompt customization interface"""
    
    st.markdown("### ✏️ Customize Prompts")
    
    # Initialize prompts in session state
    if 'custom_ai_prompt' not in st.session_state:
        st.session_state.custom_ai_prompt = get_default_ai_prompt()
    if 'custom_web_prompt' not in st.session_state:
        st.session_state.custom_web_prompt = get_default_web_prompt()
    
    # AI Prompt Editor
    with st.expander("🤖 AI Classification Prompt", expanded=True):
        ai_prompt = st.text_area(
            "Edit AI Prompt",
            value=st.session_state.custom_ai_prompt,
            height=300,
            help="Use {product_info} and {categories} as placeholders"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save AI Prompt"):
                st.session_state.custom_ai_prompt = ai_prompt
                st.success("✅ AI prompt saved!")
        with col2:
            if st.button("🔄 Reset AI Prompt"):
                st.session_state.custom_ai_prompt = get_default_ai_prompt()
                st.rerun()
    
    # Web Prompt Editor
    with st.expander("🔍 Web Search Prompt"):
        web_prompt = st.text_area(
            "Edit Web Prompt",
            value=st.session_state.custom_web_prompt,
            height=200,
            help="Use {description} and {snippets} as placeholders"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Web Prompt"):
                st.session_state.custom_web_prompt = web_prompt
                st.success("✅ Web prompt saved!")
        with col2:
            if st.button("🔄 Reset Web Prompt"):
                st.session_state.custom_web_prompt = get_default_web_prompt()
                st.rerun()
    
    if st.button("❌ Close Prompt Editor"):
        st.session_state.show_prompt_editor = False
        st.rerun()


def run_test_classification(config, settings, num_rows):
    """Run classification on a small sample"""
    
    with st.spinner(f"Testing on first {num_rows} rows..."):
        try:
            # Create temp file for test
            df = st.session_state.input_df.head(num_rows)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                df.to_excel(tmp.name, index=False)
                tmp_path = tmp.name
            
            # Get custom prompts if available
            custom_ai_prompt = st.session_state.get('custom_ai_prompt')
            custom_web_prompt = st.session_state.get('custom_web_prompt')
            
            # Create engine
            engine = ClassificationEngine(
                config,
                model_name=settings['model_name'],
                custom_ai_prompt=custom_ai_prompt,
                custom_web_prompt=custom_web_prompt
            )
            
            # Get taxonomy file if provided
            taxonomy_file = None
            if settings.get('taxonomy_file'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_tax:
                    tmp_tax.write(settings['taxonomy_file'].getvalue())
                    taxonomy_file = tmp_tax.name
            
            # Run classification
            progress_placeholder = st.empty()
            
            def progress_callback(msg):
                progress_placeholder.info(msg)
            
            output_file, project_dir, stats, result_df = engine.classify_batch(
                tmp_path,
                taxonomy_file=taxonomy_file,
                project_name=f"Test_{pd.Timestamp.now().strftime('%H%M%S')}",
                progress_callback=progress_callback,
                use_web_search=settings.get('use_web_search', False),
                save_to_disk=False  # Don't save test results to disk
            )
            
            # result_df is now returned directly (no need to read Excel file)
            
            st.success(f"✅ Test complete! Processed {num_rows} rows in {stats['duration']:.1f}s")
            
            st.markdown("### 📊 Test Results")
            st.dataframe(result_df, use_container_width=True)
            
            # Show stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("AI Processed", stats['ai_processed'])
            with col2:
                st.metric("Web Processed", stats['web_processed'])
            with col3:
                st.metric("High Confidence", stats['high_confidence'])
            with col4:
                st.metric("Cost", f"${stats['cost']:.4f}")
            
        except Exception as e:
            st.error(f"❌ Test failed: {e}")


def run_full_classification(config, settings, project_name):
    """Run full classification on entire dataset"""
    
    df = st.session_state.input_df
    
    with st.spinner("Classification in progress..."):
        try:
            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                settings['uploaded_file'].seek(0)
                tmp.write(settings['uploaded_file'].getvalue())
                tmp_path = tmp.name
            
            # Get custom prompts
            custom_ai_prompt = st.session_state.get('custom_ai_prompt')
            custom_web_prompt = st.session_state.get('custom_web_prompt')
            
            # Create engine
            engine = ClassificationEngine(
                config,
                model_name=settings['model_name'],
                custom_ai_prompt=custom_ai_prompt,
                custom_web_prompt=custom_web_prompt
            )
            
            # Get taxonomy file if provided
            taxonomy_file = None
            if settings.get('taxonomy_file'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_tax:
                    tmp_tax.write(settings['taxonomy_file'].getvalue())
                    taxonomy_file = tmp_tax.name
            
            # Progress tracking
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            def progress_callback(msg):
                progress_text.info(msg)
                # Update progress bar based on message
                if "Processed" in msg:
                    try:
                        parts = msg.split()
                        current = int(parts[1].split('/')[0])
                        total = int(parts[1].split('/')[1])
                        progress_bar.progress(current / total)
                    except:
                        pass
            
            # Run classification
            output_file, project_dir, stats, result_df = engine.classify_batch(
                tmp_path,
                taxonomy_file=taxonomy_file,
                project_name=project_name,
                progress_callback=progress_callback,
                use_web_search=settings.get('use_web_search', True)
            )
            
            progress_bar.progress(1.0)
            
            # Store results in session state (use returned DataFrame)
            st.session_state.last_results = {
                'output_file': output_file,
                'project_dir': project_dir,
                'stats': stats,
                'df': result_df
            }
            
            st.success(f"✅ Classification complete!")
            st.balloons()
            
            # Show summary
            st.markdown("### 📊 Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", stats['total'])
            with col2:
                st.metric("Duration", f"{stats['duration']:.1f}s")
            with col3:
                st.metric("Total Cost", f"${stats['cost']:.4f}")
            with col4:
                st.metric("High Confidence", stats['high_confidence'])
            
            # Download button
            with open(output_file, 'rb') as f:
                st.download_button(
                    label="📥 Download Results",
                    data=f,
                    file_name=output_file.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
            
            st.info(f"📁 Results saved to: {project_dir}")
            
        except Exception as e:
            st.error(f"❌ Classification failed: {e}")
            import traceback
            st.code(traceback.format_exc())