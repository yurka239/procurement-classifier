"""
Interactive Clustering Tool v4 - Complete Workflow
Matches the main tool's normalization logic exactly:
1. Fingerprint normalization (all columns)
2. N-gram normalization (optional, for remaining variations)
3. Category clustering (based on normalized fields)

Same output as main classification tool's normalization phase.
"""

import streamlit as st
import pandas as pd
import time
import re
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List
from difflib import SequenceMatcher
import time

st.set_page_config(
    page_title="Clustering Tool v4 - Complete",
    page_icon="⚡",
    layout="wide"
)


class FastClusterer:
    """Optimized clustering matching main tool logic"""
    
    def __init__(self, ngram_size: int = 2):
        self.ngram_size = ngram_size
        self._cache = {}
    
    def key_fingerprint(self, text: str) -> str:
        """OpenRefine-compatible fingerprint (matches main tool)"""
        if not text or pd.isna(text):
            return ""
        if text in self._cache:
            return self._cache[text]
        
        original = text
        text = str(text).strip().lower()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = sorted(set(text.split()))
        result = ' '.join(tokens)
        self._cache[original] = result
        return result
    
    def ngram_fingerprint(self, text: str) -> str:
        """N-gram fingerprint for character-level matching"""
        if not text or pd.isna(text):
            return ""
        cache_key = f"ng_{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9]', '', text)
        if len(text) < self.ngram_size:
            result = text
        else:
            ngrams = set()
            for i in range(len(text) - self.ngram_size + 1):
                ngrams.add(text[i:i + self.ngram_size])
            result = ''.join(sorted(ngrams))
        self._cache[cache_key] = result
        return result
    
    def cluster_column(self, values: pd.Series, method: str = 'fingerprint') -> Dict:
        """Cluster a single column"""
        unique_values = values.dropna().unique()
        clusters = defaultdict(list)
        
        # Debug: Log sample values
        if len(unique_values) > 0:
            print(f"DEBUG: Processing {len(unique_values)} unique values")
            print(f"DEBUG: Sample values: {list(unique_values[:5])}")
        
        for value in unique_values:
            fp = self.key_fingerprint(value) if method == 'fingerprint' else self.ngram_fingerprint(value)
            if fp:
                clusters[fp].append(value)
        
        # Debug: Log clustering results
        multi_variant_clusters = {fp: vals for fp, vals in clusters.items() if len(vals) >= 2}
        print(f"DEBUG: Total fingerprints: {len(clusters)}")
        print(f"DEBUG: Multi-variant clusters: {len(multi_variant_clusters)}")
        if len(multi_variant_clusters) > 0:
            print(f"DEBUG: Sample cluster: {list(multi_variant_clusters.items())[0]}")
        
        return multi_variant_clusters
    
    def cluster_categories(self, df: pd.DataFrame, key_cols: List[str], category_col: str) -> Dict:
        """
        Cluster categories based on user-selected key columns
        
        Args:
            df: DataFrame with data
            key_cols: List of column names to use for clustering (e.g., ['Concept_Noun', 'Modifier'])
            category_col: Column name to normalize (e.g., 'Final_Category')
        """
        # Create clustering key from selected columns
        def make_cluster_key(row):
            parts = []
            for col in key_cols:
                val = str(row.get(col, '')).strip()
                # Use fingerprint for all fields except the first one (to allow exact match on primary field)
                if len(parts) == 0:
                    parts.append(val.lower())
                else:
                    parts.append(self.key_fingerprint(val))
            return '|'.join(parts)
        
        df['_cluster_key'] = df.apply(make_cluster_key, axis=1)
        
        # Group by cluster key
        clusters = df.groupby('_cluster_key').groups
        
        # Build category clusters
        category_clusters = {}
        for cluster_key, indices in clusters.items():
            if len(indices) < 2:
                continue
            
            # Get categories in this cluster
            categories = df.loc[indices, category_col].dropna()
            categories = categories[categories.astype(str).str.strip() != '']
            
            if len(categories) > 0:
                # Find most common category
                category_counts = Counter(categories)
                most_common = category_counts.most_common(1)[0][0]
                
                # Only cluster if >50% consensus
                if category_counts[most_common] > len(categories) * 0.5:
                    category_clusters[cluster_key] = {
                        'indices': list(indices),
                        'variants': list(categories.unique()),
                        'canonical': most_common,
                        'total_rows': len(indices),
                        'key_cols': key_cols,
                        'category_col': category_col
                    }
        
        return category_clusters


def cluster_all_columns(df: pd.DataFrame, columns: List[str], method: str, threshold: float = 0.8) -> Dict:
    """Cluster multiple columns"""
    clusterer = FastClusterer()
    results = {}
    
    progress = st.progress(0)
    status = st.empty()
    
    for idx, col in enumerate(columns):
        status.text(f"Clustering {col}...")
        start = time.time()
        
        # Debug info
        unique_count = df[col].nunique()
        non_null_count = df[col].notna().sum()
        
        if method == 'fuzzy':
            clusters = fuzzy_cluster_column(df[col], threshold)
        else:
            clusters = clusterer.cluster_column(df[col], method=method)
        elapsed = time.time() - start
        
        results[col] = {
            'clusters': clusters,
            'elapsed': elapsed,
            'count': len(clusters),
            'unique_values': unique_count,
            'non_null_values': non_null_count
        }
        progress.progress((idx + 1) / len(columns))
    
    progress.empty()
    status.empty()
    return results


def fuzzy_cluster_column(values: pd.Series, threshold: float = 0.8) -> Dict:
    """Fuzzy matching clustering using string similarity"""
    unique_values = values.dropna().unique()
    clusters = defaultdict(list)
    processed = set()
    
    for i, val1 in enumerate(unique_values):
        if val1 in processed:
            continue
        
        # Start a new cluster with this value
        cluster_key = val1
        cluster_members = [val1]
        processed.add(val1)
        
        # Find similar values
        for val2 in unique_values[i+1:]:
            if val2 in processed:
                continue
            
            # Calculate similarity
            similarity = SequenceMatcher(None, str(val1).lower(), str(val2).lower()).ratio()
            
            if similarity >= threshold:
                cluster_members.append(val2)
                processed.add(val2)
        
        # Only keep clusters with 2+ members
        if len(cluster_members) >= 2:
            clusters[cluster_key] = cluster_members
    
    return dict(clusters)


def create_cluster_dataframe(column_name: str, clusters: Dict, df: pd.DataFrame) -> pd.DataFrame:
    """Create editable dataframe for clusters (only multi-variant clusters)"""
    rows = []
    
    for fingerprint, variant_list in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
        # Skip single-variant clusters (nothing to review)
        if len(variant_list) <= 1:
            continue
        
        variant_counts = {v: df[column_name].eq(v).sum() for v in variant_list}
        total_rows = sum(variant_counts.values())
        sorted_variants = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create variants display with separator for easy parsing
        variants_display = " | ".join([f"{v} ({c})" for v, c in sorted_variants])
        
        # Store individual variants for exclusion feature
        variant_names = [v for v, c in sorted_variants]
        
        default_canonical = sorted_variants[0][0]
        
        rows.append({
            'Approve': True,
            'Variants': variants_display,
            'Normalize_To': default_canonical,
            'Exclude': '',  # User can list variants to exclude, separated by semicolon
            'Total_Rows': total_rows,
            '_fingerprint': fingerprint,
            '_variant_list': variant_list
        })
    
    return pd.DataFrame(rows)


def create_category_cluster_dataframe(category_clusters: Dict, df: pd.DataFrame, category_col: str = 'Final_Category') -> pd.DataFrame:
    """Create editable dataframe for category clusters (only multi-variant clusters)"""
    rows = []
    
    for cluster_key, cluster_data in sorted(category_clusters.items(), key=lambda x: x[1]['total_rows'], reverse=True):
        # Skip single-variant clusters (nothing to review)
        if len(cluster_data['variants']) <= 1:
            continue
        
        # Get category column from cluster data or use parameter
        cat_col = cluster_data.get('category_col', category_col)
        
        # Count occurrences of each variant
        variant_counts = {}
        for idx in cluster_data['indices']:
            cat = df.loc[idx, cat_col]
            variant_counts[cat] = variant_counts.get(cat, 0) + 1
        
        # Sort by count (descending) and format with counts
        sorted_variants = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)
        variants_display = " ⚫ ".join([f"{v} ({c})" for v, c in sorted_variants])
        
        # Get actual key column names from cluster data
        key_cols = cluster_data.get('key_cols', [])
        
        # Get actual values from first row in cluster
        first_idx = cluster_data['indices'][0]
        key_values = []
        for col in key_cols:
            if col in df.columns:
                val = str(df.loc[first_idx, col])
                key_values.append(val if val != 'nan' else '')
            else:
                key_values.append('')
        
        clustering_reason = " + ".join(key_values) if key_values else cluster_key
        
        rows.append({
            'Approve': True,
            'Clustering_Key': clustering_reason,
            'Variants': variants_display,
            'Normalize_To': cluster_data['canonical'],
            'Total_Rows': cluster_data['total_rows'],
            '_cluster_key': cluster_key,
            '_indices': cluster_data['indices'],
            '_variant_counts': variant_counts
        })
    
    return pd.DataFrame(rows)


def apply_field_normalizations(df: pd.DataFrame, all_results: Dict) -> pd.DataFrame:
    """Apply field normalizations (fingerprint/n-gram) with main tool's post-processing"""
    df_normalized = df.copy()
    changes = []
    
    for column_name, result_data in all_results.items():
        if 'edited_df' not in result_data:
            continue
        
        edited_df = result_data['edited_df']
        cluster_df = result_data['cluster_df']
        
        value_mapping = {}
        for idx, row in edited_df.iterrows():
            if row['Approve']:
                canonical = row['Normalize_To']
                
                # Get exclusion list
                exclude_str = str(row.get('Exclude', '')).strip()
                excluded_variants = set()
                if exclude_str:
                    # Parse excluded variants (semicolon or comma separated)
                    excluded_variants = {v.strip() for v in re.split(r'[;,]', exclude_str) if v.strip()}
                
                # Apply main tool's post-processing rules
                if 'Concept_Noun' in column_name:
                    # Lowercase + singularize
                    canonical = canonical.lower()
                    canonical = _simple_singularize(canonical)
                elif 'Category_Type' in column_name:
                    # Lowercase + remove "category/type" words
                    canonical = canonical.lower()
                    canonical = re.sub(r'\b(category|type|class)\b', '', canonical).strip()
                    canonical = re.sub(r'\s+', ' ', canonical)
                
                variant_list = cluster_df.iloc[idx]['_variant_list']
                for variant in variant_list:
                    # Skip if variant is in exclusion list
                    if variant in excluded_variants:
                        continue
                    if variant != canonical:
                        value_mapping[variant] = canonical
        
        if value_mapping:
            df_normalized[column_name] = df_normalized[column_name].replace(value_mapping)
            changes.append(f"**{column_name}**: {len(value_mapping)} values")
    
    return df_normalized, changes


def _simple_singularize(text: str) -> str:
    """Simple singularization (matches main tool)"""
    if not text:
        return text
    
    # Common plural patterns
    if text.endswith('ies') and len(text) > 4:
        return text[:-3] + 'y'
    elif text.endswith('ves'):
        return text[:-3] + 'f'
    elif text.endswith('ses') and len(text) > 4:
        return text[:-2]
    elif text.endswith('s') and not text.endswith('ss'):
        return text[:-1]
    
    return text


@st.fragment
def render_cluster_editor(cluster_df: pd.DataFrame, col_name: str, step: str) -> pd.DataFrame:
    """Isolated data editor that only reruns this fragment, not entire app"""
    
    if len(cluster_df) > 0:
        st.info(f"✅ Showing {len(cluster_df)} clusters with 2+ variants. Single-variant clusters auto-approved.")
        
        with st.expander("💡 Tips for editing clusters", expanded=False):
            st.markdown("""
            **How to use:**
            - **Approve**: Check to apply normalization, uncheck to skip
            - **Variants**: All variations found (with counts). Click a variant name to copy it
            - **Normalize To**: Edit to change the canonical value. You can type any value or copy from Variants
            - **Exclude**: Enter variant names to exclude from normalization (separate with semicolons)
              - Example: `AAA battery; AAAA battery` will exclude these from the cluster
            - **Total_Rows**: Number of rows affected
            
            **Quick actions:**
            - To exclude a variant: Copy its name from Variants column and paste into Exclude column
            - To use a different variant as canonical: Copy from Variants and paste into Normalize To
            
            **⚡ Performance tip:** Edits in this table only refresh this section, not the entire app!
            """)
        
        # Use data_editor - now isolated in fragment!
        edited = st.data_editor(
            cluster_df[['Approve', 'Variants', 'Normalize_To', 'Exclude', 'Total_Rows']],
            column_config={
                'Approve': st.column_config.CheckboxColumn('Approve', default=True),
                'Variants': st.column_config.TextColumn('Variants (count)', disabled=True, width='large'),
                'Normalize_To': st.column_config.TextColumn('Normalize To', width='medium'),
                'Exclude': st.column_config.TextColumn('Exclude (semicolon-separated)', width='medium', help='Enter variant names to exclude from normalization'),
                'Total_Rows': st.column_config.NumberColumn('Rows', disabled=True, width='small')
            },
            hide_index=True,
            use_container_width=True,
            key=f"{step}_{col_name}",
            disabled=False
        )
        
        return edited
    else:
        st.info("No clusters to review for this column.")
        return cluster_df


def apply_category_normalizations(df: pd.DataFrame, edited_df: pd.DataFrame, cluster_df: pd.DataFrame, category_col: str = 'Final_Category') -> tuple:
    """Apply category normalizations"""
    df_normalized = df.copy()
    changes_count = 0
    
    for idx, row in edited_df.iterrows():
        if row['Approve']:
            canonical = row['Normalize_To']
            indices = cluster_df.iloc[idx]['_indices']
            
            # Update category column for all rows in cluster
            mask = df_normalized.index.isin(indices)
            old_values = df_normalized.loc[mask, category_col]
            df_normalized.loc[mask, category_col] = canonical
            changes_count += (old_values != canonical).sum()
    
    return df_normalized, changes_count


def main():
    st.title("⚡ Clustering Tool v4 - Complete Workflow")
    st.markdown("*Matches main tool: Fingerprint → N-gram → Category Clustering*")
    
    # Performance note
    with st.expander("ℹ️ About Performance", expanded=False):
        st.markdown("""
        **Data editor is instant!** Checkbox and text edits in the cluster tables update locally without reruns.
        
        - ✅ Click checkboxes in cluster tables → instant
        - ✅ Edit text in cluster tables → instant  
        - ✅ Scroll through clusters → instant
        
        **Note:** Dropdowns and multiselects (for column selection) DO trigger reruns - this is normal Streamlit behavior.
        
        **Rerun only happens when you click the green "Apply" button** at the bottom of each step.
        
        If you're experiencing slow performance:
        - Make sure you're using Streamlit 1.28+ 
        - Large datasets (50k+ rows) may take time to cluster initially
        - Cluster dataframes are cached to avoid recreation on reruns
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Upload
        st.subheader("1️⃣ Upload File")
        uploaded = st.file_uploader("File", type=['csv', 'xlsx'])
        
        if not uploaded:
            st.info("📤 Upload checkpoint.xlsx or classified output")
            st.stop()
        
        # Load
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            st.success(f"✅ {len(df):,} rows")
            
            if 'df_original' not in st.session_state:
                st.session_state['df_original'] = df
                st.session_state['filename'] = uploaded.name
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
        
        st.markdown("---")
        
        # Workflow steps
        st.subheader("📋 Workflow")
        st.markdown("""
        **Step 1:** Fingerprint (word-level)
        **Step 2:** N-gram (character-level)
        **Step 3:** Category clustering
        **Step 4:** Export
        """)
        
        st.markdown("---")
        
        # Current step indicator
        current_step = st.session_state.get('current_step', 1)
        st.info(f"**Current Step:** {current_step}")
        
        # Reset button
        if st.button("🔄 Start Over", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['df_original', 'filename']:
                    del st.session_state[key]
            st.rerun()
    
    # Main area
    df = st.session_state['df_original']
    current_step = st.session_state.get('current_step', 1)
    
    # Step 1: Fingerprint Normalization
    if current_step == 1:
        st.header("Step 1: Fingerprint Normalization (Word-Level)")
        st.info("Normalizes case, punctuation, word order. Fast and catches most variations.")
        
        # MERGE AI AND WEB OUTPUTS (match main tool)
        st.info("📋 Merging AI and Web outputs before normalization")
        
        # Merge Concept Nouns
        if 'Concept_Noun_Raw' in df.columns or 'Web_Concept_Noun' in df.columns:
            merged_nouns = []
            for idx, row in df.iterrows():
                ai_noun = str(row.get('Concept_Noun_Raw', '')).strip()
                web_noun = str(row.get('Web_Concept_Noun', '')).strip()
                # Prefer Web if available, otherwise AI
                if web_noun and web_noun.lower() != 'nan':
                    merged_nouns.append(web_noun)
                elif ai_noun and ai_noun.lower() != 'nan':
                    merged_nouns.append(ai_noun)
                else:
                    merged_nouns.append('')
            df['Concept_Noun_Merged'] = merged_nouns
            st.success(f"✅ Merged Concept_Noun: AI ({df['Concept_Noun_Raw'].notna().sum()}) + Web ({df.get('Web_Concept_Noun', pd.Series()).notna().sum()})")
        
        # Merge Category Types
        if 'Category_Type_Raw' in df.columns or 'Web_Category_Type' in df.columns:
            merged_cats = []
            for idx, row in df.iterrows():
                ai_cat = str(row.get('Category_Type_Raw', '')).strip()
                web_cat = str(row.get('Web_Category_Type', '')).strip()
                # Prefer Web if available, otherwise AI
                if web_cat and web_cat.lower() != 'nan':
                    merged_cats.append(web_cat)
                elif ai_cat and ai_cat.lower() != 'nan':
                    merged_cats.append(ai_cat)
                else:
                    merged_cats.append('')
            df['Category_Type_Merged'] = merged_cats
            st.success(f"✅ Merged Category_Type: AI ({df['Category_Type_Raw'].notna().sum()}) + Web ({df.get('Web_Category_Type', pd.Series()).notna().sum()})")
        
        # Merge Modifiers (if Web_Modifier exists)
        if 'Modifier_Raw' in df.columns or 'Web_Modifier' in df.columns:
            merged_mods = []
            for idx, row in df.iterrows():
                ai_mod = str(row.get('Modifier_Raw', '')).strip()
                web_mod = str(row.get('Web_Modifier', '')).strip()
                # Prefer Web if available, otherwise AI
                if web_mod and web_mod.lower() != 'nan':
                    merged_mods.append(web_mod)
                elif ai_mod and ai_mod.lower() != 'nan':
                    merged_mods.append(ai_mod)
                else:
                    merged_mods.append('')
            df['Modifier_Merged'] = merged_mods
            st.success(f"✅ Merged Modifier: AI ({df['Modifier_Raw'].notna().sum()}) + Web ({df.get('Web_Modifier', pd.Series()).notna().sum()})")
        
        # Update working dataframe
        st.session_state['df_original'] = df
        
        # Show ALL columns from file, not just preset ones
        available_cols = df.columns.tolist()
        
        # Suggest common columns as default (prefer merged over raw)
        suggested_defaults = []
        for col in ['Concept_Noun_Merged', 'Category_Type_Merged', 'Modifier_Merged', 'Brand_Raw',
                    'Product_Name', 'Supplier', 'Category', 'Type', 'Brand', 'Modifier']:
            if col in available_cols:
                suggested_defaults.append(col)
        
        # If no merged columns, fallback to raw columns
        if not any('Merged' in col for col in suggested_defaults):
            for col in ['Concept_Noun_Raw', 'Category_Type_Raw', 'Modifier_Raw']:
                if col in available_cols and col not in suggested_defaults:
                    suggested_defaults.append(col)
        
        # Show form only if results not yet generated
        if 'fingerprint_results' not in st.session_state:
            # Use form to prevent reruns on column selection
            with st.form("column_selection_form"):
                st.markdown("**Select columns to normalize:**")
                
                # Get previous selection or use defaults
                if 'step1_selected_cols' not in st.session_state:
                    st.session_state['step1_selected_cols'] = suggested_defaults if suggested_defaults else available_cols[:4]
                
                selected_cols = st.multiselect(
                    "Columns",
                    available_cols,
                    default=st.session_state.get('step1_selected_cols', suggested_defaults),
                    help="Choose which columns to normalize. Changes won't apply until you click the button below.",
                    label_visibility="collapsed"
                )
                
                submitted = st.form_submit_button("🔍 Run Fingerprint Clustering", type="primary", use_container_width=True)
            
            # Only process when form is submitted
            if submitted:
                # Update session state with new selection
                st.session_state['step1_selected_cols'] = selected_cols
                
                if not selected_cols:
                    st.warning("Select at least one column")
                    st.stop()
                
                with st.spinner("Clustering..."):
                    working_df = st.session_state.get('df_working', df)
                    
                    # Debug: Show data info
                    st.info(f"📊 Processing {len(working_df)} rows across {len(selected_cols)} columns")
                    for col in selected_cols:
                        unique_count = working_df[col].nunique()
                        null_count = working_df[col].isna().sum()
                        st.write(f"- **{col}**: {unique_count} unique values, {null_count} nulls")
                        # Show sample values
                        sample_vals = working_df[col].dropna().unique()[:3]
                        st.write(f"  Sample: {list(sample_vals)}")
                    
                    results = cluster_all_columns(working_df, selected_cols, 'fingerprint')
                    st.session_state['fingerprint_results'] = results
                    st.session_state['selected_cols'] = selected_cols  # Save for Step 2!
                    
                    # Pre-create cluster dataframes to avoid recreating on every rerun
                    cluster_dfs = {}
                    for col_name in selected_cols:
                        cluster_dfs[col_name] = create_cluster_dataframe(
                            col_name, 
                            results[col_name]['clusters'], 
                            working_df
                        )
                    st.session_state['fingerprint_cluster_dfs'] = cluster_dfs
                    
                    st.success("✅ Fingerprint clustering complete!")
                    st.rerun()
            else:
                # Show current selection
                if st.session_state.get('step1_selected_cols'):
                    st.info(f"Currently selected: {', '.join(st.session_state['step1_selected_cols'])}")
                st.stop()
        
        # Show results
        if 'fingerprint_results' in st.session_state and 'fingerprint_cluster_dfs' in st.session_state:
            results = st.session_state['fingerprint_results']
            cluster_dfs = st.session_state['fingerprint_cluster_dfs']
            selected_cols = st.session_state.get('selected_cols', [])  # Get from session state
            total_clusters = sum(r['count'] for r in results.values())
            
            st.metric("Total Clusters Found", total_clusters)
            
            if total_clusters > 0:
                tabs = st.tabs([f"{col} ({results[col]['count']})" for col in selected_cols])
                edited_data = {}
                
                for idx, col_name in enumerate(selected_cols):
                    with tabs[idx]:
                        # Use cached cluster dataframe (no recreation on rerun!)
                        cluster_df = cluster_dfs[col_name]
                        
                        # Use fragment for isolated reruns - MUCH faster!
                        edited = render_cluster_editor(cluster_df, col_name, "fp")
                        edited_data[col_name] = {'edited_df': edited, 'cluster_df': cluster_df}
                
                st.markdown("---")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    apply_btn = st.button("✅ Apply Fingerprint & Continue to N-gram", type="primary", use_container_width=True)
                with col2:
                    if st.button("🔄 Reset Step 1", use_container_width=True):
                        # Clear Step 1 results to restart
                        if 'fingerprint_results' in st.session_state:
                            del st.session_state['fingerprint_results']
                        if 'fingerprint_cluster_dfs' in st.session_state:
                            del st.session_state['fingerprint_cluster_dfs']
                        st.rerun()
                
                if apply_btn:
                    df_working = st.session_state.get('df_working', df).copy()
                    df_normalized, changes = apply_field_normalizations(df_working, edited_data)
                    
                    # Create output columns from normalized columns (match main tool)
                    if 'Concept_Noun_Merged' in df_normalized.columns:
                        df_normalized['Concept_Noun'] = df_normalized['Concept_Noun_Merged']
                    if 'Category_Type_Merged' in df_normalized.columns:
                        df_normalized['Category_Type'] = df_normalized['Category_Type_Merged']
                    if 'Modifier_Merged' in df_normalized.columns:
                        df_normalized['Modifier'] = df_normalized['Modifier_Merged']  # Use merged!
                    elif 'Modifier_Raw' in df_normalized.columns:
                        df_normalized['Modifier'] = df_normalized['Modifier_Raw']  # Fallback to raw
                    if 'Brand_Raw' in df_normalized.columns:
                        df_normalized['Brand'] = df_normalized['Brand_Raw']  # Now normalized!
                    
                    st.session_state['df_working'] = df_normalized
                    st.session_state['fingerprint_changes'] = changes
                    st.session_state['current_step'] = 2
                    st.success("✅ Fingerprint normalization applied!")
                    st.rerun()
    
    # Step 2: N-gram Normalization
    elif current_step == 2:
        st.header("Step 2: N-gram Normalization (Character-Level)")
        st.info("Catches spacing variations, abbreviations, typos. Works on fingerprint-normalized data.")
        
        # Show fingerprint changes
        if 'fingerprint_changes' in st.session_state:
            st.subheader("Fingerprint Changes Applied:")
            for change in st.session_state['fingerprint_changes']:
                st.markdown(f"- {change}")
        
        st.markdown("---")
        
        # Get columns from Step 1
        previous_cols = st.session_state.get('selected_cols', [])
        
        if not previous_cols:
            st.error("No columns selected in Step 1. Please go back.")
            st.stop()
        
        # Show form only if results not yet generated
        if 'ngram_results' not in st.session_state:
            # Use form to prevent reruns
            with st.form("ngram_column_selection_form"):
                st.markdown("**Select columns for N-gram clustering:**")
                
                selected_cols = st.multiselect(
                    "Columns",
                    previous_cols,
                    default=previous_cols,
                    help="Choose which columns to run N-gram clustering on. Changes won't apply until you click the button below.",
                    label_visibility="collapsed"
                )
                
                submitted = st.form_submit_button("🔍 Run N-gram Clustering", type="primary", use_container_width=True)
            
            if submitted:
                if not selected_cols:
                    st.warning("Select at least one column")
                    st.stop()
                
                with st.spinner("Clustering..."):
                    working_df = st.session_state['df_working']
                    results = cluster_all_columns(working_df, selected_cols, 'ngram')
                    st.session_state['ngram_results'] = results
                    
                    # Pre-create cluster dataframes to avoid recreating on every rerun
                    cluster_dfs = {}
                    for col_name in selected_cols:
                        cluster_dfs[col_name] = create_cluster_dataframe(
                            col_name, 
                            results[col_name]['clusters'], 
                            working_df
                        )
                    st.session_state['ngram_cluster_dfs'] = cluster_dfs
                    
                st.success("✅ N-gram clustering complete!")
                st.rerun()
            else:
                st.info(f"Currently selected: {', '.join(previous_cols)}")
                st.stop()
        
        # Show results
        if 'ngram_results' in st.session_state and 'ngram_cluster_dfs' in st.session_state:
            results = st.session_state['ngram_results']
            cluster_dfs = st.session_state['ngram_cluster_dfs']
            selected_cols = list(results.keys())  # Get from results keys
            total_clusters = sum(r['count'] for r in results.values())
            
            st.metric("Additional Clusters Found", total_clusters)
            
            if total_clusters > 0:
                tabs = st.tabs([f"{col} ({results[col]['count']})" for col in selected_cols])
                edited_data = {}
                
                for idx, col_name in enumerate(selected_cols):
                    with tabs[idx]:
                        # Use cached cluster dataframe (no recreation on rerun!)
                        cluster_df = cluster_dfs[col_name]
                        
                        # Use fragment for isolated reruns - MUCH faster!
                        edited = render_cluster_editor(cluster_df, col_name, "ng")
                        edited_data[col_name] = {'edited_df': edited, 'cluster_df': cluster_df}
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    apply_btn = st.button("✅ Apply & Continue", type="primary", use_container_width=True)
                with col2:
                    fuzzy_btn = st.button("🔍 Try Fuzzy Matching", use_container_width=True)
                with col3:
                    if st.button("🔄 Reset", use_container_width=True):
                        if 'ngram_results' in st.session_state:
                            del st.session_state['ngram_results']
                        if 'ngram_cluster_dfs' in st.session_state:
                            del st.session_state['ngram_cluster_dfs']
                        st.rerun()
                
                if apply_btn:
                    df_normalized, changes = apply_field_normalizations(st.session_state['df_working'], edited_data)
                    st.session_state['df_working'] = df_normalized
                    st.session_state['ngram_changes'] = changes
                    st.session_state['current_step'] = 3
                    st.success("✅ N-gram normalization applied!")
                    st.rerun()
                
                if fuzzy_btn:
                    # Apply current N-gram changes first
                    df_normalized, changes = apply_field_normalizations(st.session_state['df_working'], edited_data)
                    st.session_state['df_working'] = df_normalized
                    st.session_state['ngram_changes'] = changes
                    # Go to fuzzy matching step
                    st.session_state['current_step'] = 2.5
                    st.rerun()
            else:
                st.info("No additional clusters found. Fields are already well-normalized!")
                col1, col2 = st.columns([3, 1])
                with col1:
                    skip_btn = st.button("➡️ Skip to Category Clustering", type="primary", use_container_width=True)
                with col2:
                    if st.button("🔄 Reset Step 2", use_container_width=True):
                        if 'ngram_results' in st.session_state:
                            del st.session_state['ngram_results']
                        if 'ngram_cluster_dfs' in st.session_state:
                            del st.session_state['ngram_cluster_dfs']
                        st.session_state['current_step'] = 1
                        st.rerun()
                
                if skip_btn:
                    st.session_state['current_step'] = 3
                    st.rerun()
    
    # Step 2.5: Fuzzy Matching (Optional)
    elif current_step == 2.5:
        st.header("Step 2.5: Fuzzy Matching (Advanced)")
        st.warning("⚠️ **Manual Review Required**: All clusters start unchecked. Review carefully before approving!")
        st.info("🔍 Finds similar words like 'scaffold' vs 'scaffolding', 'coat' vs 'coating'. Use threshold to control sensitivity.")
        
        # Get columns from previous steps
        previous_cols = st.session_state.get('selected_cols', [])
        
        if not previous_cols:
            st.error("No columns selected in previous steps. Please go back.")
            st.stop()
        
        # Show form only if results not yet generated
        if 'fuzzy_results' not in st.session_state:
            with st.form("fuzzy_settings_form"):
                st.markdown("**Fuzzy Matching Settings:**")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_cols = st.multiselect(
                        "Columns to analyze",
                        previous_cols,
                        default=previous_cols,
                        help="Select columns to run fuzzy matching on"
                    )
                
                with col2:
                    threshold = st.slider(
                        "Similarity Threshold",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.75,
                        step=0.05,
                        help="Higher = more strict (only very similar words). Lower = more loose (catches more variations)"
                    )
                
                st.markdown("""
                **Threshold Guide:**
                - **0.90-1.00**: Very strict (e.g., 'scaffold' vs 'scaffolds')
                - **0.75-0.89**: Moderate (e.g., 'scaffold' vs 'scaffolding') ⭐ Recommended
                - **0.60-0.74**: Loose (e.g., 'coat' vs 'coating')
                - **0.50-0.59**: Very loose (may group unrelated words)
                """)
                
                submitted = st.form_submit_button("🔍 Run Fuzzy Matching", type="primary", use_container_width=True)
            
            if submitted:
                if not selected_cols:
                    st.warning("Select at least one column")
                    st.stop()
                
                with st.spinner(f"Running fuzzy matching (threshold={threshold})..."):
                    working_df = st.session_state['df_working']
                    results = cluster_all_columns(working_df, selected_cols, 'fuzzy', threshold=threshold)
                    st.session_state['fuzzy_results'] = results
                    st.session_state['fuzzy_threshold'] = threshold
                    
                    # Pre-create cluster dataframes
                    cluster_dfs = {}
                    for col_name in selected_cols:
                        cluster_df = create_cluster_dataframe(
                            col_name,
                            results[col_name]['clusters'],
                            working_df
                        )
                        # Set all to unchecked by default for manual review
                        cluster_df['Approve'] = False
                        cluster_dfs[col_name] = cluster_df
                    st.session_state['fuzzy_cluster_dfs'] = cluster_dfs
                    
                st.success("✅ Fuzzy matching complete! Review clusters below.")
                st.rerun()
            else:
                st.info(f"Currently selected: {', '.join(previous_cols)} | Threshold: 0.75")
                st.stop()
        
        # Show results
        if 'fuzzy_results' in st.session_state and 'fuzzy_cluster_dfs' in st.session_state:
            results = st.session_state['fuzzy_results']
            cluster_dfs = st.session_state['fuzzy_cluster_dfs']
            selected_cols = list(results.keys())
            threshold = st.session_state.get('fuzzy_threshold', 0.75)
            total_clusters = sum(r['count'] for r in results.values())
            
            st.metric("Fuzzy Clusters Found", total_clusters)
            st.info(f"🎯 Similarity threshold: {threshold*100:.0f}% | ⚠️ All clusters start UNCHECKED - review and approve manually")
            
            if total_clusters > 0:
                tabs = st.tabs([f"{col} ({results[col]['count']})" for col in selected_cols])
                edited_data = {}
                
                for idx, col_name in enumerate(selected_cols):
                    with tabs[idx]:
                        cluster_df = cluster_dfs[col_name]
                        
                        # Use fragment for isolated reruns
                        edited = render_cluster_editor(cluster_df, col_name, "fuzzy")
                        edited_data[col_name] = {'edited_df': edited, 'cluster_df': cluster_df}
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    apply_btn = st.button("✅ Apply Fuzzy & Continue", type="primary", use_container_width=True)
                with col2:
                    retry_btn = st.button("🔁 Try Different Threshold", use_container_width=True)
                with col3:
                    skip_btn = st.button("⏭️ Skip", use_container_width=True)
                
                if apply_btn:
                    df_normalized, changes = apply_field_normalizations(st.session_state['df_working'], edited_data)
                    st.session_state['df_working'] = df_normalized
                    st.session_state['fuzzy_changes'] = changes
                    st.session_state['current_step'] = 3
                    st.success("✅ Fuzzy matching applied!")
                    st.rerun()
                
                if retry_btn:
                    # Clear results to try again
                    if 'fuzzy_results' in st.session_state:
                        del st.session_state['fuzzy_results']
                    if 'fuzzy_cluster_dfs' in st.session_state:
                        del st.session_state['fuzzy_cluster_dfs']
                    st.rerun()
                
                if skip_btn:
                    st.session_state['current_step'] = 3
                    st.rerun()
            else:
                st.info("No fuzzy clusters found. Try lowering the threshold or skip to next step.")
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button("➡️ Skip to Category Clustering", type="primary", use_container_width=True):
                        st.session_state['current_step'] = 3
                        st.rerun()
                with col2:
                    if st.button("🔁 Retry", use_container_width=True):
                        if 'fuzzy_results' in st.session_state:
                            del st.session_state['fuzzy_results']
                        if 'fuzzy_cluster_dfs' in st.session_state:
                            del st.session_state['fuzzy_cluster_dfs']
                        st.rerun()
    
    # Step 3: Category Clustering
    elif current_step == 3:
        st.header("Step 3: Category Clustering")
        st.info("Normalizes categories based on similar field combinations")
        
        # Let user select which columns to use for clustering
        df_working = st.session_state['df_working']
        available_cols = df_working.columns.tolist()
        
        # Suggest common columns
        suggested_key_cols = []
        for col in ['Concept_Noun', 'Modifier', 'Category_Type', 'Product_Name', 'Supplier', 'Type']:
            if col in available_cols:
                suggested_key_cols.append(col)
        
        suggested_category_col = None
        for col in ['Final_Category', 'Category', 'Mapped_Category']:
            if col in available_cols:
                suggested_category_col = col
                break
        
        # Column selection in form
        st.subheader("Select Clustering Fields")
        
        # Show form only if results not yet generated
        if 'category_clusters' not in st.session_state:
            # Show previous selection if available
            prev_key_cols = st.session_state.get('category_key_cols', [])
            prev_category_col = st.session_state.get('category_col', None)
            
            if prev_key_cols or prev_category_col:
                st.info(f"📝 Previous selection: Key fields: {', '.join(prev_key_cols) if prev_key_cols else 'None'} | Category: {prev_category_col or 'None'}")
            
            with st.form("category_column_selection_form"):
                # Use previous selection as default if available, otherwise use suggested
                default_key_cols = prev_key_cols if prev_key_cols else (suggested_key_cols if suggested_key_cols else available_cols[:3])
                default_category_idx = available_cols.index(prev_category_col) if prev_category_col and prev_category_col in available_cols else (available_cols.index(suggested_category_col) if suggested_category_col else 0)
                
                key_cols = st.multiselect(
                    "Fields to use for clustering (items with same values will be grouped)",
                    available_cols,
                    default=default_key_cols,
                    help="Select 2-3 fields that identify similar items. Changes won't apply until you click the button below."
                )
                
                category_col = st.selectbox(
                    "Category field to normalize",
                    available_cols,
                    index=default_category_idx,
                    help="The category field that will be normalized based on the clustering fields"
                )
                
                submitted = st.form_submit_button("🔍 Run Category Clustering", type="primary", use_container_width=True)
            
            if submitted:
                if not key_cols:
                    st.warning("Select at least one clustering field")
                    st.stop()
                
                if not category_col:
                    st.warning("Select a category field to normalize")
                    st.stop()
                
                with st.spinner("Clustering categories..."):
                    clusterer = FastClusterer()
                    df_working = st.session_state['df_working']
                    
                    # Use user-selected columns!
                    category_clusters = clusterer.cluster_categories(df_working, key_cols, category_col)
                    st.session_state['category_clusters'] = category_clusters
                    st.session_state['category_key_cols'] = key_cols
                    st.session_state['category_col'] = category_col
                    
                    # Pre-create cluster dataframe to avoid recreating on every rerun
                    cluster_df = create_category_cluster_dataframe(category_clusters, df_working, category_col)
                    st.session_state['category_cluster_df'] = cluster_df
                    
                st.success("✅ Category clustering complete!")
                st.rerun()
            else:
                st.info("Select fields above and click the button to run clustering")
                st.stop()
        
        # Show results
        if 'category_clusters' in st.session_state and 'category_cluster_df' in st.session_state:
            category_clusters = st.session_state['category_clusters']
            cluster_df = st.session_state['category_cluster_df']
            
            st.metric("Category Clusters Found", len(category_clusters))
            
            if len(cluster_df) > 0:
                st.info(f"✅ Showing {len(cluster_df)} clusters with 2+ variants. Single-variant clusters auto-approved.")
                
                # Use data_editor - edits are instant, no reruns!
                edited = st.data_editor(
                    cluster_df[['Approve', 'Clustering_Key', 'Variants', 'Normalize_To', 'Total_Rows']],
                    column_config={
                        'Approve': st.column_config.CheckboxColumn('Approve', default=True),
                        'Clustering_Key': st.column_config.TextColumn('Clustered By (Noun + Modifier + Category)', disabled=True, width='large'),
                        'Variants': st.column_config.TextColumn('Category Variants', disabled=True, width='large'),
                        'Normalize_To': st.column_config.TextColumn('Normalize To', width='medium'),
                        'Total_Rows': st.column_config.NumberColumn('Rows', disabled=True, width='small')
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="cat_editor"
                )
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    apply_btn = st.button("✅ Apply & Go to Export", type="primary", use_container_width=True)
                with col2:
                    if st.button("� Try Different Fields", use_container_width=True):
                        # Keep current results but allow re-running with different fields
                        if 'category_clusters' in st.session_state:
                            del st.session_state['category_clusters']
                        if 'category_cluster_df' in st.session_state:
                            del st.session_state['category_cluster_df']
                        st.info("💡 Select different fields below and click 'Run Category Clustering' again")
                        st.rerun()
                with col3:
                    if st.button("🔄 Reset", use_container_width=True):
                        # Full reset - clear all Step 3 data
                        if 'category_clusters' in st.session_state:
                            del st.session_state['category_clusters']
                        if 'category_cluster_df' in st.session_state:
                            del st.session_state['category_cluster_df']
                        if 'category_key_cols' in st.session_state:
                            del st.session_state['category_key_cols']
                        if 'category_col' in st.session_state:
                            del st.session_state['category_col']
                        st.rerun()
                
                if apply_btn:
                    category_col = st.session_state.get('category_col', 'Final_Category')
                    df_normalized, changes_count = apply_category_normalizations(
                        st.session_state['df_working'], edited, cluster_df, category_col
                    )
                    st.session_state['df_final'] = df_normalized
                    st.session_state['category_changes'] = changes_count
                    st.session_state['current_step'] = 4
                    st.success(f"✅ {changes_count} categories normalized!")
                    st.rerun()
            else:
                st.info("No category clusters found. Categories are already consistent!")
                if st.button("➡️ Go to Export"):
                    st.session_state['df_final'] = st.session_state['df_working']
                    st.session_state['current_step'] = 4
                    st.rerun()
    
    # Step 4: Export
    elif current_step == 4:
        st.header("Step 4: Export Normalized Data")
        
        # Summary
        st.subheader("📊 Normalization Summary")
        
        if 'fingerprint_changes' in st.session_state:
            st.markdown("**Fingerprint Normalization:**")
            for change in st.session_state['fingerprint_changes']:
                st.markdown(f"- {change}")
        
        if 'ngram_changes' in st.session_state:
            st.markdown("**N-gram Normalization:**")
            for change in st.session_state['ngram_changes']:
                st.markdown(f"- {change}")
        
        if 'fuzzy_changes' in st.session_state:
            st.markdown("**Fuzzy Matching:**")
            for change in st.session_state['fuzzy_changes']:
                st.markdown(f"- {change}")
            threshold = st.session_state.get('fuzzy_threshold', 0.75)
            st.markdown(f"  *(Threshold: {threshold*100:.0f}%)*")
        
        if 'category_changes' in st.session_state:
            st.markdown(f"**Category Clustering:** {st.session_state['category_changes']} categories normalized")
        
        st.markdown("---")
        
        # Column selection for export
        st.subheader("📋 Select Columns to Export")
        
        df_final = st.session_state['df_final']
        all_columns = df_final.columns.tolist()
        
        # Categorize columns
        normalized_cols = [col for col in all_columns if col in ['Concept_Noun', 'Category_Type', 'Modifier', 'Final_Category', 'Brand']]
        raw_cols = [col for col in all_columns if '_Raw' in col or col.startswith('Web_')]
        other_cols = [col for col in all_columns if col not in normalized_cols and col not in raw_cols]
        
        # Default selection: normalized + other (exclude raw)
        default_selection = normalized_cols + other_cols
        
        with st.expander("💡 Column Selection Tips", expanded=False):
            st.markdown("""
            **Recommended selections:**
            - **Minimal**: Only normalized columns (Concept_Noun, Category_Type, Modifier, etc.)
            - **Standard**: Normalized + other columns (excludes raw/web columns) ⭐ Default
            - **Full**: All columns (includes raw and web columns for reference)
            
            **Benefits of excluding columns:**
            - Smaller file size
            - Cleaner output
            - Faster processing
            - Easier to read
            """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_columns = st.multiselect(
                "Columns to include in export",
                all_columns,
                default=default_selection,
                help="Select which columns to include in the exported file"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✅ Select All", use_container_width=True):
                selected_columns = all_columns
                st.rerun()
            if st.button("🎯 Minimal", use_container_width=True):
                selected_columns = normalized_cols
                st.rerun()
        
        if not selected_columns:
            st.warning("⚠️ Please select at least one column to export")
            st.stop()
        
        # Create export dataframe with selected columns
        df_export = df_final[selected_columns].copy()
        st.session_state['df_export'] = df_export
        
        # Show column stats
        col_stats = st.columns(3)
        with col_stats[0]:
            st.metric("Total Columns", len(all_columns))
        with col_stats[1]:
            st.metric("Selected", len(selected_columns))
        with col_stats[2]:
            excluded = len(all_columns) - len(selected_columns)
            st.metric("Excluded", excluded)
        
        st.markdown("---")
        
        # Preview
        st.subheader("Preview (first 20 rows)")
        st.dataframe(df_export.head(20), use_container_width=True)
        
        # Export options
        st.subheader("💾 Export Options")
        
        output_file = f"normalized_{st.session_state['filename']}"
        
        # Option 1: Save to directory
        col1, col2 = st.columns([3, 1])
        with col1:
            save_dir = st.text_input(
                "Save to directory (leave empty to download)",
                placeholder="C:\\HLDWKS_ProcureText_Studio\\dev\\modules\\Text_Classification_v2.1\\output",
                help="Enter a directory path to save the file directly, or leave empty to download"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("💾 Save to Directory", disabled=not save_dir, use_container_width=True):
                try:
                    import os
                    # Create directory if it doesn't exist
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Full output path
                    output_path = os.path.join(save_dir, output_file)
                    
                    # Save file with selected columns
                    df_to_save = st.session_state.get('df_export', st.session_state['df_final'])
                    if output_file.endswith('.csv'):
                        df_to_save.to_csv(output_path, index=False)
                    else:
                        df_to_save.to_excel(output_path, index=False)
                    
                    st.success(f"✅ Saved to: {output_path}")
                    st.info(f"📊 Exported {len(df_to_save)} rows × {len(df_to_save.columns)} columns")
                except Exception as e:
                    st.error(f"❌ Error saving file: {e}")
        
        st.markdown("---")
        
        # Option 2: Download button
        st.markdown("**Or download to your browser:**")
        
        df_to_download = st.session_state.get('df_export', st.session_state['df_final'])
        
        if output_file.endswith('.csv'):
            csv = df_to_download.to_csv(index=False)
            st.download_button(
                f"⬇️ Download CSV ({len(df_to_download.columns)} columns)",
                csv,
                output_file,
                "text/csv",
                use_container_width=True
            )
        else:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_to_download.to_excel(writer, index=False)
            st.download_button(
                f"⬇️ Download Excel ({len(df_to_download.columns)} columns)",
                output.getvalue(),
                output_file,
                use_container_width=True
            )


if __name__ == "__main__":
    main()
