"""Classification Engine v2.1 - Two-Phase Processing

Phase 1: Row-by-row AI extraction (language, brand, concept, attributes)
Phase 2: Batch normalization (fingerprinting, clustering, canonical forms)
"""

import time
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Optional, Callable, Tuple, Dict, Any, List

from src.ai_handler import AIClassifier
from src.normalizer import ProcurementNormalizer
from src.web_search import search_google, search_perplexity
from src.cost_tracker import CostTracker
from src.data_loader import load_taxonomy, load_input_data
from src.attribute_config import get_attribute_key_map, get_all_attributes


class ClassificationEngine:
    """Handles the two-phase classification workflow."""
    
    def __init__(self, config, model_name=None, custom_ai_prompt=None, custom_web_prompt=None):
        self.config = config
        self.model_name = model_name or config.default_model
        self.custom_ai_prompt = custom_ai_prompt
        self.custom_web_prompt = custom_web_prompt
        self.cost_tracker = CostTracker(config.price_table, self.model_name)
        
        # Initialize normalizer if enabled
        self.normalizer = None
        if getattr(config, 'enable_normalization', True):
            self.normalizer = ProcurementNormalizer(
                fingerprint_threshold=getattr(config, 'fingerprint_threshold', 0.85),
                ngram_size=getattr(config, 'ngram_size', 2),
                min_cluster_size=getattr(config, 'min_cluster_size', 2)
            )
            print("[Engine] Phase 2 normalization: ENABLED")
        
        # Fixed 20 attributes for all categories
        self.fixed_attributes = [
            'Material', 'Size_Dimension', 'UOM', 'Thread_Type', 
            'Pressure_Rating', 'Temperature_Rating', 'Manufacturer_Part_No',
            'Voltage', 'Power', 'Quantity_Per_Pack', 'Packaging', 
            'Processor', 'RAM', 'Storage', 'Screen_Size', 
            'Software', 'Service_Type', 'Duration', 'Scope', 'Other'
        ]
        print(f"[Engine] Fixed attributes: {len(self.fixed_attributes)} attributes for all categories")
        
        # Track failed rows
        self.failed_rows: List[Dict] = []
        
    def classify_batch(
        self, 
        input_file, 
        taxonomy_file=None, 
        project_name=None, 
        progress_callback=None, 
        use_web_search=True, 
        save_to_disk=True,
        attributes_to_extract=None,
        custom_attributes=None
    ) -> Tuple[Optional[Path], Optional[Path], Dict[str, Any]]:
        """
        Main classification pipeline with two-phase processing.
        
        Args:
            input_file: Path to input Excel file
            taxonomy_file: Optional path to taxonomy file
            project_name: Optional project name
            progress_callback: Function to call with progress updates
            use_web_search: Whether to use web search for low confidence results
            save_to_disk: Whether to save results to disk
            attributes_to_extract: Optional list of attribute names to extract from AI
            custom_attributes: Optional dict of custom attribute definitions {name: {description, examples}}
        
        Returns:
            tuple: (output_file_path, project_dir, stats_dict)
        """
        # Store custom attributes for use in AI prompts
        self.custom_attributes = custom_attributes or {}
        
        # Use user-selected attributes if provided, otherwise fall back to fixed attributes
        if attributes_to_extract and len(attributes_to_extract) > 0:
            self.active_attributes = attributes_to_extract
            print(f"[Engine] Using {len(self.active_attributes)} user-selected attributes: {self.active_attributes}")
            if self.custom_attributes:
                print(f"[Engine] Including {len(self.custom_attributes)} custom attributes: {list(self.custom_attributes.keys())}")
        else:
            self.active_attributes = self.fixed_attributes
            print(f"[Engine] Using {len(self.active_attributes)} default attributes")
        
        start_time = time.time()
        
        # Generate project name if not provided
        if project_name is None:
            project_name = f"Classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create project folder
        if save_to_disk:
            project_dir = self.config.projects_dir / project_name
            project_dir.mkdir(parents=True, exist_ok=True)
            if progress_callback:
                progress_callback(f"📁 Created project folder: {project_name}")
            
            # Copy input file to project folder
            input_copy = project_dir / "input.xlsx"
            shutil.copy2(input_file, input_copy)
        else:
            project_dir = self.config.projects_dir / project_name
        
        # Load taxonomy if provided
        categories = None
        if taxonomy_file and Path(taxonomy_file).exists():
            categories = load_taxonomy(taxonomy_file)
            if save_to_disk:
                taxonomy_copy = project_dir / "taxonomy.xlsx"
                shutil.copy2(taxonomy_file, taxonomy_copy)
            if progress_callback:
                progress_callback(f"📋 Loaded {len(categories)} categories from taxonomy")
        
        # Load input data
        df = load_input_data(input_file)
        
        if progress_callback:
            progress_callback(f"📊 Loaded {len(df)} rows")
        
        # Validate required columns
        if 'Description' not in df.columns:
            raise ValueError("Input file must have 'Description' column")
        
        # Add Index column if not exists
        if 'Index' not in df.columns:
            df.insert(0, 'Index', range(1, len(df) + 1))
        
        # === PHASE 1: AI Classification ===
        if progress_callback:
            progress_callback("🔄 Phase 1: AI Classification")
        
        df = self._phase1_classification(
            df, categories, use_web_search, progress_callback, save_to_disk, project_dir
        )
        
        # === PHASE 2: Normalization ===
        if self.normalizer and progress_callback:
            progress_callback("🔄 Phase 2: Batch Normalization")
        
        if self.normalizer:
            df = self._phase2_normalization(df, progress_callback)
        
        # === PHASE 3: Generate Normalized Descriptions ===
        if self.normalizer and getattr(self.config, 'generate_normalized_descriptions', True):
            if progress_callback:
                progress_callback("🔄 Phase 3: Generating Normalized Descriptions")
            df = self._phase3_descriptions(df)
        
        # Calculate statistics
        elapsed = time.time() - start_time
        cost_summary = self.cost_tracker.get_summary()
        
        stats = {
            'total_rows': len(df),
            'elapsed_time': elapsed,
            'cost': cost_summary['total_cost'],
            'total_requests': cost_summary['total_requests'],
            'failed_rows': len(self.failed_rows)
        }
        
        if self.normalizer:
            norm_summary = self.normalizer.get_summary()
            stats.update({
                'unique_nouns': norm_summary['unique_nouns'],
                'unique_categories': norm_summary['unique_categories'],
                'unique_brands': norm_summary['unique_brands'],
                'nouns_clustered': norm_summary['nouns_clustered'],
                'categories_clustered': norm_summary['categories_clustered'],
                'brands_clustered': norm_summary['brands_clustered']
            })
        
        # Save results
        output_file = None
        if save_to_disk:
            # Select and reorder columns for output
            df_output = self._select_output_columns(df, use_web_search)
            
            output_file = project_dir / "classified_output.xlsx"
            df_output.to_excel(output_file, index=False)
            
            # Save failed rows if any
            if self.failed_rows:
                failed_df = pd.DataFrame(self.failed_rows)
                failed_file = project_dir / "failed_rows.xlsx"
                failed_df.to_excel(failed_file, index=False)
                if progress_callback:
                    progress_callback(f"⚠️ Saved {len(self.failed_rows)} failed rows to failed_rows.xlsx")
            
            # Save normalization patterns if normalizer enabled
            if self.normalizer:
                patterns_file = project_dir / "normalization_patterns.json"
                self.normalizer.save_patterns(patterns_file)
            
            if progress_callback:
                progress_callback(f"✅ Classification complete!")
                progress_callback(f"   Time: {elapsed:.1f}s | Cost: ${cost_summary['total_cost']:.4f}")
                if self.normalizer:
                    progress_callback(f"   Nouns: {stats['unique_nouns']} unique ({stats['nouns_clustered']} clustered)")
                    progress_callback(f"   Categories: {stats['unique_categories']} unique")
                    progress_callback(f"   Brands: {stats['unique_brands']} unique")
        
        return output_file, project_dir, stats
    
    def _phase1_classification(
        self,
        df: pd.DataFrame,
        categories: Optional[List[str]],
        use_web_search: bool,
        progress_callback: Optional[Callable],
        save_to_disk: bool,
        project_dir: Path
    ) -> pd.DataFrame:
        """Phase 1: Row-by-row AI classification."""
        
        # Track start time for progress estimates
        phase_start_time = time.time()
        
        # Add output columns
        output_cols = [
            'Language',
            'Translated_Description',
            'Brand_Raw',
            'Concept_Noun_Raw',
            'Modifier_Raw',
            'Category_Type_Raw',
            'Is_Product_or_Service',
            'Service_Object',
            'AI_Insight',
            'AI_Confidence',
            'Mapped_Category_Raw',
            'Proposed_Category_Raw',
            'Matching_Reason'
        ]
        
        # Add web search columns if enabled
        if use_web_search:
            output_cols.extend([
                'Web_Concept_Noun',
                'Web_Category_Type',
                'Web_Confidence',
                'Web_Mapped_Category',
                'Web_Proposed_Category',
                'Web_Matching_Reason'
            ])
        
        # Add final decision columns
        output_cols.extend([
            'Final_Category',
            'Final_Proposed_Category',
            'Classification_Method'
        ])
        
        # Add attribute columns if extraction enabled - only selected attributes
        if getattr(self.config, 'extract_attributes', True):
            # Use user-selected attributes or fall back to all fixed attributes
            attr_cols = getattr(self, 'active_attributes', self.fixed_attributes)
            output_cols.extend(attr_cols)
            # Also add custom attribute columns
            custom_attrs = getattr(self, 'custom_attributes', {})
            if custom_attrs:
                output_cols.extend(custom_attrs.keys())
            print(f"[Engine] Added {len(attr_cols)} selected attribute columns + {len(custom_attrs)} custom")
        
        for col in output_cols:
            if col not in df.columns:
                df[col] = ''
        
        # Initialize AI classifier
        ai_classifier = AIClassifier(
            api_key=self.config.openai_key,
            model_name=self.model_name,
            verbosity=getattr(self.config, 'verbosity', 'low'),
            reasoning_effort=getattr(self.config, 'reasoning_effort', 'medium'),
            extract_attributes=getattr(self.config, 'extract_attributes', True)
        )
        
        # Thread-safe processing
        lock = threading.Lock()
        processed_count = [0]
        
        def process_row(idx: int, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
            """Process a single row."""
            try:
                # Skip if already processed
                if pd.notna(row.get('Concept_Noun_Raw')) and row.get('Concept_Noun_Raw'):
                    print(f"[Debug] Row {idx} skipped - already processed")
                    return idx, {'skipped': True}
                
                # Get description
                text = str(row['Description'])
                if not text or text.lower() == 'nan':
                    print(f"[Debug] Row {idx} skipped - empty description")
                    return idx, {'skipped': True, 'reason': 'empty'}
                
                # Get additional fields
                additional = {}
                for col in df.columns:
                    if col not in ['Index', 'Description'] and pd.notna(row[col]):
                        additional[col] = str(row[col])
                
                # Debug: Show we're about to classify
                if idx < 3:  # Only show first 3 to avoid spam
                    print(f"[Debug] Row {idx} - Calling AI for: {text[:50]}...")
                
                # Classify with selected attributes (including custom)
                result, usage = ai_classifier.classify(
                    text, 
                    additional_fields=additional,
                    taxonomy=categories,
                    custom_prompt=self.custom_ai_prompt,
                    attributes_to_extract=self.active_attributes,
                    custom_attributes=getattr(self, 'custom_attributes', {})
                )
                
                # Debug: Show result
                if idx < 3:
                    print(f"[Debug] Row {idx} - AI returned: concept_noun={result.get('concept_noun', 'EMPTY')}, confidence={result.get('confidence', 'EMPTY')}")
                
                # Track cost
                if usage:
                    if idx < 3:  # Debug first 3 rows
                        print(f"[Debug] Row {idx} - Usage object: {usage}")
                        print(f"[Debug] Row {idx} - Usage type: {type(usage)}")
                        if hasattr(usage, '__dict__'):
                            print(f"[Debug] Row {idx} - Usage attributes: {usage.__dict__}")
                    self.cost_tracker.add_usage(usage)
                    if idx < 3:
                        print(f"[Debug] Row {idx} - Cost tracker: tokens={self.cost_tracker.prompt_tokens + self.cost_tracker.completion_tokens}, cost=${self.cost_tracker.total_cost():.6f}")
                else:
                    print(f"[Debug] Row {idx} - WARNING: No usage data returned!")
                
                # Store AI result
                ai_result = result.copy()
                
                # Check if web search needed (only for low confidence)
                web_result = None
                ai_confidence = result.get('confidence', 'low').lower()
                
                # Debug web search logic
                if idx < 3:
                    print(f"[Debug] Row {idx} - AI confidence: {ai_confidence}, use_web_search: {use_web_search}")
                
                if use_web_search and ai_confidence == 'low':
                    if idx < 3:
                        print(f"[Debug] Row {idx} - Web search TRIGGERED")
                    
                    provider = getattr(self.config, 'web_search_provider', 'google')
                    web_context = None
                    
                    if provider == 'google' and self.config.google_api_key:
                        web_context = search_google(text, self.config.google_api_key, self.config.google_cse_id)
                        if idx < 3:
                            print(f"[Debug] Row {idx} - Google search returned: {len(web_context) if web_context else 0} chars")
                    elif provider == 'perplexity' and self.config.perplexity_key:
                        web_context = search_perplexity(text, self.config.perplexity_key)
                    
                    if web_context:
                        if idx < 3:
                            print(f"[Debug] Row {idx} - Calling AI with web context...")
                        web_result, usage2 = ai_classifier.classify(
                            text, additional, categories, web_context, self.custom_ai_prompt,
                            attributes_to_extract=self.active_attributes,
                            custom_attributes=getattr(self, 'custom_attributes', {})
                        )
                        if idx < 3:
                            print(f"[Debug] Row {idx} - Web AI returned: concept_noun={web_result.get('concept_noun', 'EMPTY')}, confidence={web_result.get('confidence', 'EMPTY')}")
                        self.cost_tracker.add_usage(usage2)
                    else:
                        if idx < 3:
                            print(f"[Debug] Row {idx} - No web context, skipping web AI call")
                else:
                    if idx < 3:
                        print(f"[Debug] Row {idx} - Web search NOT triggered (confidence={ai_confidence})")
                
                # Determine Final Category based on confidence logic
                final_category = ''
                final_proposed = ''
                method = ''
                
                web_confidence = web_result.get('confidence', '').lower() if web_result else ''
                ai_mapped = ai_result.get('mapped_category', '')
                ai_proposed = ai_result.get('proposed_category', '')
                web_mapped = web_result.get('mapped_category', '') if web_result else ''
                web_proposed = web_result.get('proposed_category', '') if web_result else ''
                
                # Decision logic: Prefer matched categories over proposed
                # 1. If Web has matched category with high confidence → use it
                # 2. If AI has matched category with high/medium confidence → use it
                # 3. If Web has matched category (any confidence) → use it
                # 4. If no matched category exists → use proposed category
                
                if web_confidence == 'high' and web_mapped:
                    final_category = web_mapped
                    final_proposed = ''
                    method = 'Web'
                elif ai_confidence in ['high', 'medium'] and ai_mapped:
                    final_category = ai_mapped
                    final_proposed = ''
                    method = 'AI'
                elif web_mapped:  # Web has match but lower confidence
                    final_category = web_mapped
                    final_proposed = ''
                    method = 'Web'
                else:
                    # No matched category - use proposed
                    final_category = ''
                    final_proposed = ai_proposed if ai_proposed else (web_proposed if web_proposed else 'UNCATEGORIZED')
                    method = 'Proposed' if final_proposed != 'UNCATEGORIZED' else 'No Match'
                
                # Combine results
                combined_result = {
                    'ai': ai_result,
                    'web': web_result,
                    'final_category': final_category,
                    'final_proposed': final_proposed,
                    'method': method
                }
                
                # Debug final result
                if idx < 3:
                    print(f"[Debug] Row {idx} - Final: category='{final_category}', proposed='{final_proposed}', method='{method}'")
                    print(f"[Debug] Row {idx} - Web result exists: {web_result is not None}")
                
                return idx, combined_result
                
            except Exception as e:
                print(f"[Debug] Row {idx} - EXCEPTION: {str(e)}")
                import traceback
                print(f"[Debug] Traceback: {traceback.format_exc()}")
                error_info = {
                    'index': idx,
                    'description': text[:100] if 'text' in locals() else '',
                    'error': str(e)
                }
                with lock:
                    self.failed_rows.append(error_info)
                return idx, {'error': str(e), 'skipped': True}
        
        # Prepare rows for processing
        rows_to_process = [(i, row) for i, row in df.iterrows()]
        
        # Process in parallel
        max_workers = getattr(self.config, 'max_workers', 10)
        batch_size = getattr(self.config, 'batch_size', 50)
        checkpoint_every = getattr(self.config, 'checkpoint_every', 100)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, i, r): i for i, r in rows_to_process}
            
            for future in as_completed(futures):
                idx, result = future.result()
                
                with lock:
                    if not result.get('skipped'):
                        # Debug: Show result structure
                        if idx < 3:
                            print(f"[Debug] Row {idx} - Result keys: {list(result.keys())}")
                        
                        # Extract AI and Web results from combined result
                        ai_res = result.get('ai', {})
                        web_res = result.get('web')
                        
                        # Debug: Check if ai_res is empty
                        if idx < 3:
                            print(f"[Debug] Row {idx} - AI result keys: {list(ai_res.keys()) if ai_res else 'EMPTY'}")
                            print(f"[Debug] Row {idx} - Web result: {'EXISTS' if web_res else 'NONE'}")
                        
                        # AI columns
                        df.at[idx, 'Language'] = ai_res.get('language', '')
                        df.at[idx, 'Translated_Description'] = ai_res.get('translated_text', '')
                        brand_value = ai_res.get('brand', '')
                        # Debug: Show brand value
                        if idx < 5:
                            print(f"[Debug] Row {idx} - Brand from AI result: '{brand_value}' (type: {type(brand_value)})")
                        df.at[idx, 'Brand_Raw'] = brand_value
                        # Verify it was set correctly
                        if idx < 5:
                            actual_value = df.at[idx, 'Brand_Raw']
                            print(f"[Debug] Row {idx} - Brand_Raw after setting: '{actual_value}'")
                            if brand_value != actual_value:
                                print(f"[ALERT] Row {idx} - MISMATCH! Set '{brand_value}' but got '{actual_value}'")
                        concept_noun = ai_res.get('concept_noun', '')
                        df.at[idx, 'Concept_Noun_Raw'] = concept_noun
                        
                        # Debug: Verify it was set
                        if idx < 3:
                            print(f"[Debug] Row {idx} - Set Concept_Noun_Raw to: '{concept_noun}'")
                        
                        df.at[idx, 'Modifier_Raw'] = ai_res.get('modifier', '')
                        df.at[idx, 'Category_Type_Raw'] = ai_res.get('category_type', '')
                        df.at[idx, 'Is_Product_or_Service'] = ai_res.get('is_product_or_service', 'Product')
                        df.at[idx, 'Service_Object'] = ai_res.get('service_object', '')
                        df.at[idx, 'AI_Insight'] = ai_res.get('ai_insight', '')
                        df.at[idx, 'AI_Confidence'] = ai_res.get('confidence', 'low')
                        df.at[idx, 'Mapped_Category_Raw'] = ai_res.get('mapped_category', '')
                        df.at[idx, 'Proposed_Category_Raw'] = ai_res.get('proposed_category', '')
                        df.at[idx, 'Matching_Reason'] = ai_res.get('matching_reason', '')
                        
                        # Web search columns (if enabled and web result exists)
                        if use_web_search and web_res:
                            if idx < 3:
                                print(f"[Debug] Row {idx} - Populating web columns")
                            df.at[idx, 'Web_Concept_Noun'] = web_res.get('concept_noun', '')
                            df.at[idx, 'Web_Category_Type'] = web_res.get('category_type', '')
                            df.at[idx, 'Web_Confidence'] = web_res.get('confidence', '')
                            df.at[idx, 'Web_Mapped_Category'] = web_res.get('mapped_category', '')
                            df.at[idx, 'Web_Proposed_Category'] = web_res.get('proposed_category', '')
                            df.at[idx, 'Web_Matching_Reason'] = web_res.get('matching_reason', '')
                        elif idx < 3:
                            print(f"[Debug] Row {idx} - NOT populating web columns (use_web_search={use_web_search}, web_res exists={web_res is not None})")
                        
                        # Final decision columns
                        final_cat = result.get('final_category', '')
                        final_prop = result.get('final_proposed', '')
                        class_method = result.get('method', '')
                        df.at[idx, 'Final_Category'] = final_cat
                        df.at[idx, 'Final_Proposed_Category'] = final_prop
                        df.at[idx, 'Classification_Method'] = class_method
                        
                        if idx < 3:
                            print(f"[Debug] Row {idx} - Set Final_Category='{final_cat}', Final_Proposed='{final_prop}', Method='{class_method}'")
                        
                        # Add attributes if present - only for selected attributes
                        if 'attributes' in ai_res and getattr(self.config, 'extract_attributes', True):
                            attrs = ai_res['attributes']
                            
                            # Only populate attributes that user selected
                            active_attrs = getattr(self, 'active_attributes', self.fixed_attributes)
                            
                            # Get dynamic key mapping from attribute_config.py
                            attr_key_map = get_attribute_key_map(active_attrs)
                            
                            # Only populate selected attributes
                            for attr_name in active_attrs:
                                key = attr_key_map.get(attr_name, attr_name.lower().replace(' ', '_'))
                                df.at[idx, attr_name] = attrs.get(key, '')
                            
                            if idx < 3:
                                print(f"[Debug] Row {idx} - Populated {len(active_attrs)} selected attributes: {active_attrs}")
                    
                    processed_count[0] += 1
                    
                    # Progress update with time estimates and cost
                    if progress_callback and processed_count[0] % batch_size == 0:
                        elapsed = time.time() - phase_start_time
                        remaining = len(rows_to_process) - processed_count[0]
                        eta_seconds = (elapsed / processed_count[0]) * remaining if processed_count[0] > 0 else 0
                        
                        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
                        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                        
                        # Get current cost
                        cost_summary = self.cost_tracker.get_summary()
                        current_cost = cost_summary['total_cost']
                        
                        progress_callback(f"⏳ Processed {processed_count[0]}/{len(rows_to_process)} rows | "
                                        f"Elapsed: {elapsed_str} | ETA: {eta_str} | Cost: ${current_cost:.4f}")
                    
                    # Checkpoint - single file that gets overwritten (not multiple files)
                    if save_to_disk and checkpoint_every > 0 and processed_count[0] % checkpoint_every == 0:
                        # Ensure directory exists before saving
                        project_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_file = project_dir / "checkpoint.xlsx"  # Single file, overwrite each time
                        df.to_excel(checkpoint_file, index=False)
                        if progress_callback:
                            progress_callback(f"💾 Checkpoint saved at row {processed_count[0]}")
        
        # Save final checkpoint after loop completes (captures last batch up to 99 rows)
        if save_to_disk and checkpoint_every > 0:
            # Ensure directory exists before saving
            project_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = project_dir / "checkpoint.xlsx"
            df.to_excel(checkpoint_file, index=False)
            if progress_callback:
                progress_callback(f"💾 Final checkpoint saved: {processed_count[0]} rows")
        
        if progress_callback:
            progress_callback(f"   Phase 1 complete: {processed_count[0]} rows processed")
        
        return df
    
    def _phase2_normalization(
        self,
        df: pd.DataFrame,
        progress_callback: Optional[Callable]
    ) -> pd.DataFrame:
        """Phase 2: Batch normalization using fingerprinting."""
        
        # Add normalized columns
        df['Concept_Noun'] = ''
        df['Modifier'] = ''
        df['Category_Type'] = ''
        df['Brand'] = ''
        df['Mapped_Category'] = ''
        
        # Note: No need to create normalized attribute columns
        # Dynamic attributes are already in Attribute_1 through Attribute_N columns
        
        # MERGE AI AND WEB OUTPUTS: Combine both sources as variants for clustering
        if progress_callback:
            progress_callback("   Merging AI and Web outputs...")
        
        # Merge Concept Nouns (AI + Web)
        merged_nouns = []
        for idx, row in df.iterrows():
            ai_noun = str(row.get('Concept_Noun_Raw', '')).strip()
            web_noun = str(row.get('Web_Concept_Noun', '')).strip()
            # Prefer non-empty value; if both exist, clustering will decide
            if web_noun and web_noun.lower() != 'nan':
                merged_nouns.append(web_noun)
            elif ai_noun and ai_noun.lower() != 'nan':
                merged_nouns.append(ai_noun)
            else:
                merged_nouns.append('')
        
        # Merge Category Types (AI + Web)
        merged_categories = []
        for idx, row in df.iterrows():
            ai_cat = str(row.get('Category_Type_Raw', '')).strip()
            web_cat = str(row.get('Web_Category_Type', '')).strip()
            # Prefer non-empty value; if both exist, clustering will decide
            if web_cat and web_cat.lower() != 'nan':
                merged_categories.append(web_cat)
            elif ai_cat and ai_cat.lower() != 'nan':
                merged_categories.append(ai_cat)
            else:
                merged_categories.append('')
        
        # Normalize Concept Nouns (using merged AI+Web values)
        if progress_callback:
            progress_callback("   Normalizing concept nouns (AI+Web merged)...")
        
        _, normalized_nouns = self.normalizer.normalize_column(merged_nouns, 'noun')
        df['Concept_Noun'] = normalized_nouns
        
        # Normalize Category Types (using merged AI+Web values)
        if progress_callback:
            progress_callback("   Normalizing categories (AI+Web merged)...")
        
        _, normalized_categories = self.normalizer.normalize_column(merged_categories, 'category')
        df['Category_Type'] = normalized_categories
        
        # Normalize Brands (clustering for consistency)
        if progress_callback:
            progress_callback("   Normalizing brands...")
        
        # Skip normalization for brands - just uppercase them directly
        # Normalization was corrupting short brand codes like 'MSA' -> 'm'
        df['Brand'] = df['Brand_Raw'].fillna('').apply(lambda x: str(x).strip().upper())
        
        # Merge Modifiers (AI + Web)
        if progress_callback:
            progress_callback("   Merging modifiers (AI+Web)...")
        
        merged_modifiers = []
        for idx, row in df.iterrows():
            ai_mod = str(row.get('Modifier_Raw', '')).strip()
            web_mod = str(row.get('Web_Modifier', '')).strip()
            # Prefer Web if available, otherwise use AI
            if web_mod and web_mod.lower() != 'nan':
                merged_modifiers.append(web_mod)
            elif ai_mod and ai_mod.lower() != 'nan':
                merged_modifiers.append(ai_mod)
            else:
                merged_modifiers.append('')
        
        # Normalize Modifiers (using merged AI+Web values)
        if progress_callback:
            progress_callback("   Normalizing modifiers (AI+Web merged)...")
        
        _, normalized_modifiers = self.normalizer.normalize_column(merged_modifiers, 'generic')
        df['Modifier'] = normalized_modifiers
        
        # Note: Attribute normalization skipped for dynamic attributes
        # Dynamic attributes are already in "Name: Value" format in Attribute_1 through Attribute_N columns
        # If specific attribute normalization is needed, it can be added here by parsing the dynamic columns
        
        # Copy mapped category and proposed category
        df['Mapped_Category'] = df['Mapped_Category_Raw']
        df['Proposed_Category'] = df['Proposed_Category_Raw']
        
        # Normalize Final_Category for consistency
        # Group items with same concept_noun + modifier (or category_type if modifier blank)
        # and apply the most common Final_Category to all items in the group
        if progress_callback:
            progress_callback("   Normalizing category consistency...")
        
        df = self._normalize_category_consistency(df)
        
        if progress_callback:
            progress_callback("   Phase 2 complete")
        
        return df
    
    def _normalize_category_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OPTIMIZED: Normalize Final_Category and Mapped_Category for consistency across similar items.
        
        Uses fast vectorized groupby instead of slow nested loops.
        
        Logic:
        1. Group by Concept_Noun (exact match)
        2. Within each group, use fingerprinting for modifier/category_type
        3. Find most common category in each cluster (>50% threshold)
        4. Apply that category to all items in the cluster
        """
        from collections import Counter
        import re
        import unicodedata
        
        # OpenRefine-style fingerprint function (inline for speed)
        def fingerprint(text):
            """
            OpenRefine-compatible key collision fingerprint:
            1. Strip whitespace
            2. Lowercase
            3. Normalize unicode (remove accents)
            4. Remove punctuation (replace with space)
            5. Split, sort unique tokens, rejoin
            """
            if not text or pd.isna(text):
                return ""
            # Strip and lowercase
            text = str(text).strip().lower()
            # Normalize unicode
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))
            # Remove punctuation, replace with space
            text = re.sub(r'[^\w\s]', ' ', text)
            # Collapse multiple spaces
            text = re.sub(r'\s+', ' ', text)
            # Split, sort unique tokens, rejoin
            tokens = sorted(set(text.split()))
            return ' '.join(tokens)
        
        # Pre-compute fingerprints (vectorized - much faster!)
        df['_modifier_fp'] = df['Modifier'].fillna('').apply(fingerprint)
        df['_category_type_fp'] = df['Category_Type'].fillna('').apply(fingerprint)
        
        # Create clustering key: concept_noun + modifier_fingerprint (or category_type if modifier empty)
        df['_cluster_key'] = df.apply(
            lambda row: f"{str(row['Concept_Noun']).lower().strip()}|{row['_modifier_fp'] if row['_modifier_fp'] else row['_category_type_fp']}", 
            axis=1
        )
        
        # Group by cluster key (FAST - no nested loops!)
        clusters = df.groupby('_cluster_key').groups
        
        # Normalize categories within each cluster (OPTIMIZED - vectorized operations)
        changes_final = 0
        changes_mapped = 0
        total_clusters = len(clusters)
        multi_item_clusters = sum(1 for indices in clusters.values() if len(indices) >= 2)
        
        print(f"[Normalizer] Category consistency: Found {total_clusters} total clusters, {multi_item_clusters} with 2+ items")
        
        # Process clusters using vectorized operations (much faster than df.at[])
        for cluster_key, indices in clusters.items():
            if len(indices) < 2:
                continue
            
            # Get cluster data (vectorized - single operation)
            cluster_data = df.loc[indices, ['Final_Category', 'Mapped_Category']]
            
            # Normalize Final_Category
            final_cats = cluster_data['Final_Category'].dropna()
            final_cats = final_cats[final_cats.astype(str).str.strip() != '']
            
            if len(final_cats) > 0:
                category_counts = Counter(final_cats)
                most_common_cat, most_common_count = category_counts.most_common(1)[0]
                
                # Apply if >50% consensus
                threshold = len(final_cats) * 0.5
                if most_common_count > threshold or len(category_counts) == 1:
                    # Vectorized update (much faster than loop with df.at[])
                    mask = (df.index.isin(indices)) & (df['Final_Category'] != most_common_cat)
                    changes_final += mask.sum()
                    df.loc[mask, 'Final_Category'] = most_common_cat
            
            # Normalize Mapped_Category
            mapped_cats = cluster_data['Mapped_Category'].dropna()
            mapped_cats = mapped_cats[mapped_cats.astype(str).str.strip() != '']
            
            if len(mapped_cats) > 0:
                category_counts = Counter(mapped_cats)
                most_common_cat, most_common_count = category_counts.most_common(1)[0]
                
                # Apply if >50% consensus
                threshold = len(mapped_cats) * 0.5
                if most_common_count > threshold or len(category_counts) == 1:
                    # Vectorized update
                    mask = (df.index.isin(indices)) & (df['Mapped_Category'] != most_common_cat)
                    changes_mapped += mask.sum()
                    df.loc[mask, 'Mapped_Category'] = most_common_cat
        
        print(f"[Normalizer] Category consistency: {changes_final} Final_Category + {changes_mapped} Mapped_Category normalized across {multi_item_clusters} clusters")
        
        # Clean up temporary columns
        df.drop(columns=['_modifier_fp', '_category_type_fp', '_cluster_key'], inplace=True)
        
        return df
    
    def _phase3_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 3: Generate normalized descriptions."""
        
        df['Normalized_Description'] = ''
        df['Confidence_Score'] = 0
        df['Needs_Review'] = 0
        
        for idx, row in df.iterrows():
            # Build attributes dict
            attributes = {}
            if getattr(self.config, 'extract_attributes', True):
                attributes = {
                    'size_primary': row.get('Size_Primary', ''),
                    'material': row.get('Material', ''),
                    'material_grade': row.get('Material_Grade', ''),
                    'color': row.get('Color', ''),
                    'thread_type': row.get('Thread_Type', '')
                }
            
            # Generate normalized description
            original_text = str(row.get('Translated_Description') or row.get('Description', ''))
            norm_desc = self.normalizer.generate_normalized_description(
                concept_noun=row.get('Concept_Noun', ''),
                original_text=original_text,
                brand=row.get('Brand', ''),
                modifier=row.get('Modifier', ''),
                attributes=attributes
            )
            df.at[idx, 'Normalized_Description'] = norm_desc
            
            # Calculate confidence score
            score = self._calculate_confidence_score(row)
            df.at[idx, 'Confidence_Score'] = score
            df.at[idx, 'Needs_Review'] = 1 if score < 70 else 0
        
        return df
    
    def _calculate_confidence_score(self, row: pd.Series) -> int:
        """
        Calculate a numeric confidence score (0-100).
        
        Factors:
        - AI confidence (base)
        - Brand found (+5)
        - Category mapped to taxonomy (+5)
        - Concept noun not empty (+10)
        - Description length (penalty if too short)
        """
        # Base score from AI confidence
        ai_conf = str(row.get('AI_Confidence', 'low')).lower()
        score = {'high': 85, 'medium': 65, 'low': 45}.get(ai_conf, 45)
        
        # Adjustments
        if row.get('Brand'):
            score += 5
        
        if row.get('Mapped_Category'):
            score += 5
        
        if row.get('Concept_Noun'):
            score += 5
        else:
            score -= 15
        
        if row.get('Category_Type'):
            score += 3
        
        # Penalty for very short descriptions
        original = str(row.get('Description', ''))
        if len(original) < 10:
            score -= 10
        
        # Cap score
        return max(0, min(100, score))
    
    def _select_output_columns(self, df: pd.DataFrame, use_web_search: bool) -> pd.DataFrame:
        """Select and reorder columns for final output.
        
        Removes Raw and Web intermediate columns, keeps only final normalized columns.
        """
        # Rename Final_Category to Category and create Proposed_Category and Classification_Source
        if 'Final_Category' in df.columns:
            df['Category'] = df['Final_Category']
            # Show Proposed_Category only when Category is empty
            df['Proposed_Category'] = df.apply(
                lambda row: row.get('Final_Proposed_Category', '') if not row.get('Final_Category', '') else '',
                axis=1
            )
            # Rename Classification_Method to Classification_Source
            if 'Classification_Method' in df.columns:
                df['Classification_Source'] = df['Classification_Method']
        
        # FINAL CLEANUP: Fix Brand column (uppercase, reject single chars)
        if 'Brand' in df.columns:
            df['Brand'] = df['Brand'].apply(lambda x: str(x).strip().upper() if pd.notna(x) and len(str(x).strip()) > 1 else '')
        
        # Essential columns (always include)
        output_cols = [
            'Index',
            'Description',
            'Language',
            'Translated_Description',
            'Brand',
            'Concept_Noun',
            'Modifier',
            'Category_Type',
            'Is_Product_or_Service',
            'Service_Object',
            'AI_Insight',
            'AI_Confidence',
            'Category',  # Final category (renamed from Final_Category)
            'Proposed_Category',  # Only shown when Category is empty
            'Classification_Source'  # AI/Web/Proposed (renamed from Classification_Method)
        ]
        
        # Add fixed attribute columns
        if getattr(self.config, 'extract_attributes', True):
            attr_cols = [
                'Material', 'Size_Dimension', 'UOM', 'Thread_Type',
                'Pressure_Rating', 'Temperature_Rating', 'Voltage', 'Power',
                'Quantity_Per_Pack', 'Packaging', 'Model', 'Processor',
                'RAM', 'Storage', 'Screen_Size', 'Software', 'Service_Type',
                'Duration', 'Scope', 'Other'
            ]
            output_cols.extend([col for col in attr_cols if col in df.columns])
        
        # Add normalized description and confidence if available
        if 'Normalized_Description' in df.columns:
            output_cols.extend(['Normalized_Description', 'Confidence_Score', 'Needs_Review'])
        
        # Add any additional columns from input (like Supplier, Total, etc.)
        # Exclude Raw, Web, Final_* intermediate columns
        excluded_patterns = ['_Raw', 'Web_', 'Final_', 'Mapped_Category', 'Matching_Reason', 'Classification_Method']
        for col in df.columns:
            if col not in output_cols:
                # Check if column should be excluded
                should_exclude = any(pattern in col for pattern in excluded_patterns)
                if not should_exclude:
                    output_cols.append(col)
        
        # Filter to only columns that exist in df
        final_cols = [col for col in output_cols if col in df.columns]
        
        return df[final_cols]
    
    def normalize_checkpoint(
        self,
        checkpoint_file: str,
        output_file: str = None,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Run Phase 2 and Phase 3 normalization on a checkpoint file.
        
        Use this to:
        - Resume interrupted runs
        - Re-normalize results with updated settings
        - Clean up inconsistent categories
        
        Args:
            checkpoint_file: Path to checkpoint.xlsx file
            output_file: Optional path for output (default: checkpoint_normalized.xlsx)
            progress_callback: Progress update function
        
        Returns:
            Normalized DataFrame
        """
        if progress_callback:
            progress_callback(f"📂 Loading checkpoint: {checkpoint_file}")
        
        # Load checkpoint
        df = pd.read_excel(checkpoint_file)
        
        if progress_callback:
            progress_callback(f"📊 Loaded {len(df)} rows from checkpoint")
        
        # === PHASE 2: Normalization ===
        if self.normalizer:
            if progress_callback:
                progress_callback("🔄 Phase 2: Batch Normalization")
            df = self._phase2_normalization(df, progress_callback)
        
        # === PHASE 3: Generate Normalized Descriptions ===
        if self.normalizer and getattr(self.config, 'generate_normalized_descriptions', True):
            if progress_callback:
                progress_callback("🔄 Phase 3: Generating Normalized Descriptions")
            df = self._phase3_descriptions(df)
        
        # Save normalized output
        if output_file is None:
            checkpoint_path = Path(checkpoint_file)
            output_file = checkpoint_path.parent / f"{checkpoint_path.stem}_normalized.xlsx"
        
        df.to_excel(output_file, index=False)
        
        if progress_callback:
            progress_callback(f"✅ Normalized checkpoint saved: {output_file}")
        
        return df
