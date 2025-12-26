"""
Results Display Tab
"""

import streamlit as st
import pandas as pd
from src.ui.utils import format_confidence

def render_results_tab():
    """Render the results viewing interface"""
    
    st.header("📊 Classification Results")
    
    if 'last_results' not in st.session_state:
        st.info("No results available yet. Run a classification first.")
        return
    
    results = st.session_state.last_results
    df = results['df']
    stats = results['stats']
    
    # Summary metrics
    st.markdown("### 📈 Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Rows", stats['total'])
    with col2:
        st.metric("AI Processed", stats['ai_processed'])
    with col3:
        st.metric("Web Processed", stats['web_processed'])
    with col4:
        st.metric("Duration", f"{stats['duration']:.1f}s")
    with col5:
        st.metric("Cost", f"${stats['cost']:.4f}")
    
    st.markdown("---")
    
    # Confidence breakdown
    st.markdown("### 🎯 Confidence Distribution")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 High", stats['high_confidence'])
    with col2:
        st.metric("🟡 Medium", stats['medium_confidence'])
    with col3:
        st.metric("🔴 Low", stats['low_confidence'])
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔍 Filter Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_filter = st.multiselect(
            "Confidence Level",
            options=['high', 'medium', 'low', 'none'],
            default=['high', 'medium', 'low', 'none']
        )
    
    with col2:
        if 'Category_Type' in df.columns:
            category_types = df['Category_Type'].dropna().unique().tolist()
            category_filter = st.multiselect(
                "Category Type",
                options=['All'] + category_types,
                default=['All']
            )
    
    with col3:
        search_term = st.text_input("Search Description", "")
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'AI_Confidence' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['AI_Confidence'].str.lower().isin(confidence_filter)]
    
    if 'Category_Type' in filtered_df.columns and 'All' not in category_filter:
        filtered_df = filtered_df[filtered_df['Category_Type'].isin(category_filter)]
    
    if search_term:
        if 'Description' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Description'].str.contains(search_term, case=False, na=False)]
    
    st.caption(f"Showing {len(filtered_df)} of {len(df)} rows")
    
    st.markdown("---")
    
    # Results table
    st.markdown("### 📋 Detailed Results")
    
    # Select columns to display
    display_cols = st.multiselect(
        "Select columns to display",
        options=df.columns.tolist(),
        default=[col for col in ['Description', 'Compact_Noun', 'Category_Type', 'AI_Insight', 'AI_Confidence', 'Final_Category'] if col in df.columns]
    )
    
    if display_cols:
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            height=500
        )
    
    st.markdown("---")
    
    # Download options
    st.markdown("### 📥 Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download filtered results
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Filtered Results (CSV)",
            data=csv,
            file_name="filtered_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download full results
        if results.get('output_file'):
            with open(results['output_file'], 'rb') as f:
                st.download_button(
                    label="📊 Download Full Results (Excel)",
                    data=f,
                    file_name=results['output_file'].name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    st.markdown("---")
    
    # Individual row inspection
    st.markdown("### 🔬 Inspect Individual Row")
    
    row_idx = st.number_input(
        "Select row index",
        min_value=0,
        max_value=len(filtered_df)-1,
        value=0
    )
    
    if row_idx < len(filtered_df):
        row = filtered_df.iloc[row_idx]
        
        with st.expander("📝 Row Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Input:**")
                for col in df.columns:
                    if col not in ['Compact_Noun', 'Category_Type', 'AI_Insight', 'AI_Confidence', 
                                   'AI_Category', 'AI_Proposed_Category', 'Web_Compact_Noun', 
                                   'Web_Category_Type', 'Web_Insight', 'Web_Confidence',
                                   'Web_Category', 'Web_Proposed_Category', 'Final_Category', 
                                   'Classification_Method']:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            st.text(f"{col}: {row[col]}")
            
            with col2:
                st.markdown("**AI Results:**")
                if 'Compact_Noun' in row:
                    st.text(f"Compact Noun: {row.get('Compact_Noun', '')}")
                if 'Category_Type' in row:
                    st.text(f"Category Type: {row.get('Category_Type', '')}")
                if 'AI_Insight' in row:
                    st.text(f"Insight: {row.get('AI_Insight', '')}")
                if 'AI_Confidence' in row:
                    st.text(f"Confidence: {format_confidence(row.get('AI_Confidence', ''))}")
                if 'Final_Category' in row:
                    st.text(f"Final Category: {row.get('Final_Category', '')}")
                if 'Classification_Method' in row:
                    st.text(f"Method: {row.get('Classification_Method', '')}")