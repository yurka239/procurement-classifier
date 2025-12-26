"""AI Handler for Procurement Classifier v2.1

Supports both GPT-5 (Responses API) and GPT-4 (Chat Completions API).
Extracts structured attributes dynamically based on user selection.
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI
from .attribute_config import build_prompt_section, get_attribute_info, get_attribute_key_map


class AIClassifier:
    """Handles AI API calls for classification with structured attribute extraction."""
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gpt-5-mini",
        verbosity: str = "low",
        reasoning_effort: str = "medium",
        extract_attributes: bool = True
    ):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.verbosity = verbosity
        self.reasoning_effort = reasoning_effort
        self.extract_attributes = extract_attributes
        self.is_gpt5 = model_name.startswith("gpt-5")
        
        print(f"[AI] Initialized {model_name} ({'GPT-5 API' if self.is_gpt5 else 'GPT-4 API'})")
        if extract_attributes:
            print(f"[AI] Structured attribute extraction: ENABLED")
    
    def _build_dynamic_json_template(self, attributes_to_extract: Optional[List[str]], taxonomy: bool) -> str:
        """Build JSON template with only the selected attributes."""
        # Build attributes JSON section dynamically
        if self.extract_attributes and attributes_to_extract:
            # Get key mapping for selected attributes
            key_map = get_attribute_key_map(attributes_to_extract)
            attr_lines = []
            for attr_name in attributes_to_extract:
                key = key_map.get(attr_name, attr_name.lower().replace(' ', '_'))
                attr_lines.append(f'    "{key}": "..."')
            attrs_json = ',\n'.join(attr_lines)
            attrs_section = f''',
  "attributes": {{
{attrs_json}
  }}'''
        else:
            attrs_section = ''
        
        taxonomy_section = ''
        if taxonomy:
            taxonomy_section = ''',
  "mapped_category": "<exact match from taxonomy or empty>",
  "proposed_category": "<suggested new category or empty>",
  "matching_reason": "<explanation>"'''
        
        return f'''{{
  "language": "<2-letter code>",
  "translated_text": "<English translation or original if English>",
  "brand": "<FULL brand name like MSA, SKF, 3M or empty>",
  "concept_noun": "<1-2 word core noun only>",
  "modifier": "<comma-separated modifiers or empty>",
  "category_type": "<functional category based on purpose>",
  "is_product_or_service": "Product|Service",
  "service_object": "<equipment type or empty>"{attrs_section},
  "ai_insight": "<brief description>",
  "confidence": "high|medium|low"{taxonomy_section}
}}'''
    
    def classify(
        self, 
        text: str, 
        additional_fields: Optional[Dict[str, str]] = None,
        taxonomy: Optional[List[str]] = None,
        web_context: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        attributes_to_extract: Optional[List[str]] = None,
        use_hierarchical: bool = True,
        custom_attributes: Optional[Dict[str, Dict]] = None
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        """
        Classify a procurement line item with structured attribute extraction.
        
        Args:
            text: The product/service description
            additional_fields: Optional dict of extra columns (Supplier, Part Number, etc.)
            taxonomy: Optional list of valid categories to map to (format: "Level1|Level2")
            web_context: Optional web search results for enhancement
            custom_prompt: Optional custom prompt override
            attributes_to_extract: Optional list of attribute names to extract (e.g., ['Material', 'Size', 'Thread_Type'])
            use_hierarchical: If True and taxonomy has hierarchy, use 2-step classification
            custom_attributes: Optional dict of custom attribute definitions {name: {description, examples}}
        
        Returns:
            Tuple of (classification_result, token_usage)
        """
        # DISABLED: Hierarchical classification was causing matching issues
        # Using single-pass classification (like v1.1) for better accuracy
        # if use_hierarchical and taxonomy and any('|' in cat for cat in taxonomy):
        #     return self._classify_hierarchical(text, additional_fields, taxonomy, web_context, custom_prompt, attributes_to_extract)
        
        # Standard single-pass classification (proven approach from v1.1)
        if self.is_gpt5:
            return self._classify_gpt5(text, additional_fields, taxonomy, web_context, custom_prompt, attributes_to_extract, custom_attributes)
        else:
            return self._classify_gpt4(text, additional_fields, taxonomy, web_context, custom_prompt, attributes_to_extract, custom_attributes)
    
    def _classify_hierarchical(
        self,
        text: str,
        additional_fields: Optional[Dict[str, str]],
        taxonomy: List[str],
        web_context: Optional[str],
        custom_prompt: Optional[str],
        attributes_to_extract: Optional[List[str]]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        """
        Multi-level hierarchical classification (supports 1-5 levels).
        
        Process:
        1. Detect number of levels in taxonomy
        2. Classify level-by-level from top to bottom
        3. Each step narrows down options to next level
        
        Examples:
        - 2 levels: "MRO|MRO Parts"
        - 3 levels: "MRO|Bearings|Ball Bearings"
        - 4 levels: "MRO|Bearings|Ball Bearings|Deep Groove"
        """
        # Detect maximum depth
        max_depth = max(cat.count('|') + 1 for cat in taxonomy)
        
        # Build hierarchical tree
        tree = self._build_hierarchy_tree(taxonomy, max_depth)
        
        # Debug: Show tree structure for first item
        if text and len(text) < 100:
            print(f"[Hierarchy Debug] Max depth: {max_depth}")
            print(f"[Hierarchy Debug] Level 1 options: {list(tree.keys())[:5]}...")
        
        # Step 1: Classify to first level + extract attributes
        level1_options = list(tree.keys())
        result, usage = self._classify_to_level(
            text, additional_fields, level1_options, web_context, 
            attributes_to_extract, level=1, is_first=True
        )
        
        total_usage = usage or {'input_tokens': 0, 'output_tokens': 0}
        selected_level1 = result.get('selected_category', '')
        selected_path = [selected_level1]
        
        # Debug: Show what AI returned
        print(f"[Hierarchy Debug] AI returned selected_category: '{selected_level1}'")
        print(f"[Hierarchy Debug] AI returned concept_noun: '{result.get('concept_noun', '')}'")
        print(f"[Hierarchy Debug] AI returned confidence: '{result.get('confidence', '')}'")
        
        # Step 2-N: Drill down through remaining levels
        current_tree = tree
        for level in range(2, max_depth + 1):
            # Get the category selected at previous level
            prev_selection = selected_path[-1]
            
            if not prev_selection or prev_selection not in current_tree:
                # No match at previous level - stop drilling
                break
            
            # Get options at current level
            current_options = current_tree[prev_selection]
            
            if text and len(text) < 100:
                print(f"[Hierarchy Debug] Level {level} options type: {type(current_options)}")
                if isinstance(current_options, dict):
                    print(f"[Hierarchy Debug] Level {level} dict keys: {list(current_options.keys())[:5]}...")
                elif isinstance(current_options, list):
                    print(f"[Hierarchy Debug] Level {level} list items: {current_options[:3]}...")
            
            if not current_options:
                # No subcategories - we've reached the leaf
                if text and len(text) < 100:
                    print(f"[Hierarchy Debug] No subcategories at level {level}, stopping")
                break
            
            # Check if this is actually a leaf node disguised as a dict
            # This happens when taxonomy has entries like "Facilities|Facilities"
            # which creates tree["Facilities"]["Facilities"] = ["Facilities|Facilities"]
            if isinstance(current_options, dict):
                # If dict has only one key and it matches the parent, treat as leaf
                dict_keys = list(current_options.keys())
                if len(dict_keys) == 1 and dict_keys[0] == prev_selection:
                    # This is a leaf node - get the actual category list
                    current_options = current_options[dict_keys[0]]
                    if text and len(text) < 100:
                        print(f"[Hierarchy Debug] Detected leaf node (duplicate key '{dict_keys[0]}'), treating as final level")
            
            # If current_options are strings (leaf nodes), we're at final level
            if isinstance(current_options, list):
                # Final level - match to specific category
                if text and len(text) < 100:
                    print(f"[Hierarchy Debug] Final level {level}, matching to specific category")
                
                level_result, level_usage = self._classify_to_level(
                    text, additional_fields, current_options, web_context,
                    None, level=level, is_first=False
                )
                
                # Update usage
                if level_usage:
                    total_usage['input_tokens'] += level_usage.get('input_tokens', 0)
                    total_usage['output_tokens'] += level_usage.get('output_tokens', 0)
                
                # Store final match
                result['mapped_category'] = level_result.get('mapped_category', '')
                # IMPORTANT: Don't use AI's proposed_category - build it ourselves to ensure 2-level format
                # result['proposed_category'] = level_result.get('proposed_category', '')  # AI invents multi-level paths!
                result['matching_reason'] = ' → '.join(selected_path) + f" → {level_result.get('matching_reason', '')}"
                
                if text and len(text) < 100:
                    print(f"[Hierarchy Debug] Final mapped_category: '{result['mapped_category']}'")
                    print(f"[Hierarchy Debug] Final proposed_category: '{result['proposed_category']}'")
                
                break
            else:
                # Still have more levels - continue drilling
                level_result, level_usage = self._classify_to_level(
                    text, additional_fields, list(current_options.keys()), web_context,
                    None, level=level, is_first=False
                )
                
                # Update usage
                if level_usage:
                    total_usage['input_tokens'] += level_usage.get('input_tokens', 0)
                    total_usage['output_tokens'] += level_usage.get('output_tokens', 0)
                
                selected = level_result.get('selected_category', '')
                selected_path.append(selected)
                current_tree = current_options
        
        # If no final match, suggest new category
        if not result.get('mapped_category'):
            result['mapped_category'] = ''
            # Build proposed category with proper Level1 prefix
            # Use category_type (not concept_noun) to match taxonomy style (e.g., "Electrical", "Mechanical")
            valid_path = [p for p in selected_path if p]  # Remove empty strings
            category_suggestion = result.get('category_type', result.get('concept_noun', 'Unknown'))
            # Title case each word for consistency with taxonomy format
            category_suggestion = ' '.join(word.capitalize() for word in category_suggestion.split())
            
            if valid_path:
                result['proposed_category'] = '|'.join(valid_path) + '|' + category_suggestion
            else:
                result['proposed_category'] = category_suggestion
            result['matching_reason'] = f"Partial match: {' → '.join(valid_path) if valid_path else 'No match'}"
        
        return result, total_usage
    
    def _build_hierarchy_tree(self, taxonomy: List[str], max_depth: int) -> dict:
        """
        Build a nested dictionary tree from flat taxonomy list.
        
        Example:
        Input: ["MRO|Bearings|Ball Bearings", "MRO|Bearings|Roller Bearings", "MRO|Valves"]
        Output: {
            "MRO": {
                "Bearings": ["MRO|Bearings|Ball Bearings", "MRO|Bearings|Roller Bearings"],
                "Valves": ["MRO|Valves"]
            }
        }
        """
        tree = {}
        
        for cat in taxonomy:
            parts = cat.split('|')
            current = tree
            
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Last part - store full category path
            last_part = parts[-1]
            if last_part not in current:
                current[last_part] = []
            current[last_part].append(cat)
        
        return tree
    
    def _classify_to_level(
        self,
        text: str,
        additional_fields: Optional[Dict[str, str]],
        options: List[str],
        web_context: Optional[str],
        attributes_to_extract: Optional[List[str]],
        level: int,
        is_first: bool
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        """
        Classify to a specific level in the hierarchy.
        
        Args:
            level: Current level number (1, 2, 3, etc.)
            is_first: If True, extract all attributes; if False, only category
        """
        # Build context
        additional_context = ""
        if additional_fields:
            context_lines = []
            for key, value in additional_fields.items():
                if value and str(value).strip() and str(value).lower() != 'nan':
                    context_lines.append(f"{key}: {value}")
            if context_lines:
                additional_context = "\n\nADDITIONAL CONTEXT:\n" + "\n".join(context_lines)
        
        web_section = ""
        if web_context:
            web_section = f"\n\nWEB SEARCH CONTEXT:\n{web_context}"
        
        options_list = "\n".join(f"  • {opt}" for opt in options)
        
        # Build prompt based on whether this is first level or not
        if is_first:
            # First level: extract everything + pick category
            prompt = self._build_first_level_prompt(text, additional_context, web_section, options_list, attributes_to_extract, level)
        else:
            # Subsequent levels: just pick category
            prompt = self._build_subsequent_level_prompt(text, additional_context, web_section, options_list, level)
        
        # Call AI
        system_msg = "You are an expert procurement classification system. Respond with valid JSON only."
        
        try:
            # Note: Some models (like GPT-4o) don't support temperature=0, so we omit it
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message.content.strip()
            usage = {
                'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                'output_tokens': response.usage.completion_tokens if response.usage else 0
            }
            
            # Debug: Show AI response for first few items
            if level == 1 and text and len(text) < 100:
                print(f"[Hierarchy Debug] Level {level} AI response (first 500 chars): {content[:500]}...")
            
            if is_first:
                result = self._parse_response(content)
            else:
                result = self._parse_level_response(content)
            
            return result, usage
            
        except Exception as e:
            print(f"[AI Error] Level {level} classification failed: {e}")
            return self._empty_result() if is_first else {'selected_category': '', 'mapped_category': '', 'proposed_category': '', 'matching_reason': ''}, None
    
    def _build_prompt(
        self, 
        text: str, 
        additional_fields: Optional[Dict[str, str]],
        taxonomy: Optional[List[str]],
        web_context: Optional[str],
        custom_prompt: Optional[str],
        attributes_to_extract: Optional[List[str]] = None,
        custom_attributes: Optional[Dict[str, Dict]] = None
    ) -> str:
        """Build the classification prompt with attribute extraction."""
        
        if custom_prompt:
            return custom_prompt
        
        # Build input section
        input_section = f"Description: {text}"
        
        # Add additional context fields if provided
        additional_context = ""
        if additional_fields:
            context_lines = []
            for key, value in additional_fields.items():
                if value and str(value).strip() and str(value).lower() != 'nan':
                    context_lines.append(f"{key}: {value}")
            
            if context_lines:
                additional_context = "\n\nADDITIONAL CONTEXT (use these fields to improve classification):\n" + "\n".join(context_lines)
        
        # Build taxonomy section - provide FULL taxonomy for accurate matching
        taxonomy_section = ""
        if taxonomy and len(taxonomy) > 0:
            taxonomy_section = f"""
VALID CATEGORIES (select exact match from this list):
{chr(10).join(f'  • {cat}' for cat in taxonomy)}

IMPORTANT: You MUST use the EXACT category name from the list above (preserve pipes | and capitalization).
If no good match exists, leave mapped_category empty and suggest a new category in proposed_category.
"""
        
        # Build web context section
        web_section = ""
        if web_context:
            web_section = f"""
WEB SEARCH CONTEXT:
{web_context}
"""
        
        # Build attributes section - use dynamic attributes from attribute_config.py
        attributes_section = ""
        if self.extract_attributes and attributes_to_extract and len(attributes_to_extract) > 0:
            # Use build_prompt_section from attribute_config.py for rich descriptions
            # Pass custom_attributes to include user-defined attributes in the prompt
            attributes_section = "\n" + build_prompt_section(attributes_to_extract, custom_attributes)
        
        prompt = f"""You are an expert procurement classification system.

INPUT:
{input_section}{additional_context}{web_section}
{taxonomy_section}
TASK: Extract structured information and classify the item.

EXTRACTION RULES (extract in order):
1. **Language**: Detect language (2-letter ISO code: en, es, de, fr, it, nl, pt, etc.)
2. **Translated_Text**: Full English translation (if not already English)
3. **Brand**: Product/equipment brand ONLY if EXPLICITLY mentioned in description. Common MRO brands include: SKF, MSA, 3M, Parker, Siemens, ABB, Honeywell, Brady, Fluke, DeWalt, Milwaukee, Bosch, etc.
   - Return the COMPLETE brand name exactly as commonly known (e.g., "MSA" not "m", "3M" not "3", "SKF" not "s")
   - Short brand codes ARE valid: MSA, 3M, ABB, SKF, etc. - return them in full
   - Leave EMPTY if no brand is clearly stated
   - DO NOT guess brands - only extract if explicitly written in the text
   - DO NOT use supplier/vendor name as brand

4. **Concept_Noun**: The CORE product/service noun (1-2 words ONLY)
   PATTERN RECOGNITION:
   - Identify the PRIMARY noun that defines what this item IS
   - Strip all modifiers, specifications, and attributes
   - Use lowercase, singular form
   - Examples:
     * "Stainless steel ball valve 2 inch NPT" → "ball valve"
     * "SKF 6205-2RS deep groove bearing" → "bearing"
     * "Blue nitrile safety gloves size L" → "glove"
     * "Flange nut 7/8-9 black oxide" → "flange nut"
     * "Cap screw Grade 8 3/4-10 x 4" → "cap screw"
   - For compound items, keep 2 words max: "ball valve", "safety glove", "flange nut"

5. **Modifier**: CONCISE descriptive attributes (2-4 words MAX)
   - Extract ONLY the most important qualifiers that describe the TYPE or MATERIAL
   - DO NOT include: model numbers, sizes, technical specs (those go in attributes)
   - Use lowercase, comma-separated
   - Examples:
     * "SKF 6205-2RS deep groove bearing sealed" → "deep groove, sealed"
     * "Stainless steel ball valve 2 inch NPT" → "ball, stainless steel"
     * "Submersible slurry pump 5kW" → "submersible, slurry"
     * "Graphite gasket DN100 PN40 reinforced" → "graphite, reinforced"
     * "4-core armored cable 6mm²" → "armored, 4-core"
   - Empty if no meaningful type/material modifiers found

6. **Category_Type**: Functional category based on USE/PURPOSE (2-4 words max)
   CLASSIFICATION LOGIC:
   - Ask: "What is this item's PRIMARY FUNCTION or PURPOSE?"
   - Use industry-standard functional categories
   - Examples:
     * ball valve → "flow control" (function: controls fluid flow)
     * bearing → "rotating equipment" (function: enables rotation)
     * safety glove → "hand protection" (function: protects hands)
     * flange nut, cap screw, clevis pin → "fastener" (function: joins/secures parts)
   - Be consistent: same function = same category

7. **Is_Product_or_Service**: "Product" or "Service"
8. **Service_Object**: If service, what equipment/asset it applies to (e.g., "compressor", "forklift"). Empty if product.
9. **AI_Insight**: Brief explanation of what this item is (10-20 words)
10. **Confidence**: Your confidence level: "high", "medium", or "low"
   - high: Clear, unambiguous item with recognizable terms
   - medium: Some ambiguity but reasonable classification
   - low: Unclear, abbreviated, or insufficient information
{attributes_section}{'11' if self.extract_attributes else '10'}. **Mapped_Category**: Exact match from Valid Categories list (or empty if no match)
{'12' if self.extract_attributes else '11'}. **Proposed_Category**: Suggested category if Mapped_Category is empty
{'13' if self.extract_attributes else '12'}. **Matching_Reason**: Brief explanation (1 sentence)

RESPOND WITH JSON ONLY:
{self._build_dynamic_json_template(attributes_to_extract, taxonomy)}"""
        
        return prompt
    
    def _build_first_level_prompt(self, text: str, additional_context: str, web_section: str, 
                                   options_list: str, attributes_to_extract: Optional[List[str]], level: int) -> str:
        """Build prompt for first level classification (includes all extraction)."""
        
        attributes_section = ""
        if self.extract_attributes and attributes_to_extract:
            # Use dynamic attributes from attribute_config.py
            attributes_section = "\n" + build_prompt_section(attributes_to_extract)
        
        return f"""You are an expert procurement classification system.

INPUT:
Description: {text}{additional_context}{web_section}

LEVEL {level} CATEGORIES:
{options_list}

TASK: Extract all information AND select ONE Level {level} category from the list above.

IMPORTANT: 
- You MUST select EXACTLY ONE category from the Level {level} list above
- Do NOT create your own categories or sub-categories
- Do NOT use multi-level paths like "MRO|Mechanical|Maintenance"
- ONLY use the exact category names shown in the Level {level} list
- For Level 1: Select simple names like "MRO", "Lab", "Business Consulting", etc.

EXTRACTION RULES:
1. **Language**: 2-letter ISO code
2. **Translated_Text**: English translation
3. **Brand**: Product/equipment brand ONLY if EXPLICITLY mentioned. Short codes like MSA, 3M, ABB, SKF are valid - return complete name (e.g., "MSA" not "m"). Leave EMPTY if not stated. DO NOT guess.
4. **Concept_Noun**: Core noun (1-2 words, lowercase)
5. **Modifier**: Key attributes (2-4 words, lowercase)
6. **Category_Type**: Functional category (2-4 words, title case)
7. **Is_Product_or_Service**: "Product" or "Service"
8. **Service_Object**: Equipment type (if service)
9. **AI_Insight**: Brief description (10-20 words)
10. **Confidence**: "high", "medium", or "low"
{attributes_section}
11. **Selected_Category**: MUST be EXACTLY ONE category from the Level {level} list (e.g., "MRO" or "Lab" or "Business Consulting")

RESPOND WITH JSON ONLY:
{{
  "language": "<code>",
  "translated_text": "<text>",
  "brand": "<FULL brand like MSA, 3M, SKF or empty>",
  "concept_noun": "<noun>",
  "modifier": "<modifiers or empty>",
  "category_type": "<functional category>",
  "is_product_or_service": "Product|Service",
  "service_object": "<equipment or empty>",
  "ai_insight": "<description>",
  "confidence": "high|medium|low",
  "selected_category": "<Level {level} category>",
  "attributes": {{
    "material": "...", "size_dimension": "...", "uom": "...", "thread_type": "...",
    "pressure_rating": "...", "temperature_rating": "...", "voltage": "...", "power": "...",
    "quantity_per_pack": "...", "packaging": "...", "model": "...", "processor": "...",
    "ram": "...", "storage": "...", "screen_size": "...", "software": "...",
    "service_type": "...", "duration": "...", "scope": "...", "other": "..."
  }}
}}"""
    
    def _build_subsequent_level_prompt(self, text: str, additional_context: str, web_section: str, 
                                       options_list: str, level: int) -> str:
        """Build prompt for subsequent level classification (category only)."""
        
        return f"""You are an expert procurement classification system.

INPUT:
Description: {text}{additional_context}{web_section}

LEVEL {level} CATEGORIES:
{options_list}

TASK: Select the BEST Level {level} category OR suggest new category if no match.

MATCHING RULES:
1. **Mapped_Category**: Select best match from list above (or empty if no match)
2. **Proposed_Category**: Suggest new category if Mapped_Category is empty
3. **Matching_Reason**: Explain your decision (1-2 sentences)
4. **Selected_Category**: The category you selected (for drilling down to next level)

RESPOND WITH JSON ONLY:
{{
  "selected_category": "<category for next level>",
  "mapped_category": "<exact match or empty>",
  "proposed_category": "<suggestion or empty>",
  "matching_reason": "<explanation>"
}}"""
    
    def _parse_level_response(self, content: str) -> Dict[str, Any]:
        """Parse response from subsequent level classification."""
        result = {'selected_category': '', 'mapped_category': '', 'proposed_category': '', 'matching_reason': ''}
        
        try:
            import re
            import json
            
            # Extract JSON
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
            start = content.find("{")
            end = content.rfind("}")
            
            if start != -1 and end != -1:
                json_str = content[start:end + 1]
                obj = json.loads(json_str)
                
                result['selected_category'] = str(obj.get('selected_category', '')).strip()
                result['mapped_category'] = str(obj.get('mapped_category', '')).strip()
                result['proposed_category'] = str(obj.get('proposed_category', '')).strip()
                result['matching_reason'] = str(obj.get('matching_reason', '')).strip()
        
        except Exception as e:
            print(f"[Parse Error] Level response parsing failed: {e}")
        
        return result
    
    def _classify_gpt5(
        self, 
        text: str, 
        additional_fields: Optional[Dict[str, str]],
        taxonomy: Optional[List[str]],
        web_context: Optional[str],
        custom_prompt: Optional[str],
        attributes_to_extract: Optional[List[str]] = None,
        custom_attributes: Optional[Dict[str, Dict]] = None
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        """GPT-5 classification using Responses API."""
        
        prompt = self._build_prompt(text, additional_fields, taxonomy, web_context, custom_prompt, attributes_to_extract, custom_attributes)
        
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                text={"verbosity": self.verbosity},
                reasoning={"effort": self.reasoning_effort}
            )
            
            # Extract text from response
            output_text = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text
            
            usage = {
                'input_tokens': response.usage.input_tokens if response.usage else 0,
                'output_tokens': response.usage.output_tokens if response.usage else 0
            }
            
            result = self._parse_response(output_text)
            return result, usage
            
        except Exception as e:
            print(f"[AI Error] GPT-5 call failed: {e}")
            import traceback
            print(f"[AI Error] Full traceback: {traceback.format_exc()}")
            print(f"[AI Error] Model: {self.model_name}")
            print(f"[AI Error] Prompt length: {len(prompt) if 'prompt' in locals() else 'N/A'}")
            return self._empty_result(), None
    
    def _classify_gpt4(
        self, 
        text: str, 
        additional_fields: Optional[Dict[str, str]],
        taxonomy: Optional[List[str]],
        web_context: Optional[str],
        custom_prompt: Optional[str],
        attributes_to_extract: Optional[List[str]] = None,
        custom_attributes: Optional[Dict[str, Dict]] = None
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        """GPT-4 classification using Chat Completions API."""
        
        prompt = self._build_prompt(text, additional_fields, taxonomy, web_context, custom_prompt, attributes_to_extract, custom_attributes)
        
        system_msg = """You are an expert procurement classification system. 
Extract structured information accurately. Always respond with valid JSON only."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message.content.strip()
            usage = {
                'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                'output_tokens': response.usage.completion_tokens if response.usage else 0
            }
            
            result = self._parse_response(content)
            return result, usage
            
        except Exception as e:
            print(f"[AI Error] GPT-4 call failed: {e}")
            import traceback
            print(f"[AI Error] Full traceback: {traceback.format_exc()}")
            print(f"[AI Error] Model: {self.model_name}")
            return self._empty_result(), None
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse AI response JSON."""
        result = self._empty_result()
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
            
            # Find JSON object
            start = content.find("{")
            end = content.rfind("}")
            
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end + 1]
                obj = json.loads(json_str)
                
                # Map core fields
                result['language'] = str(obj.get('language', '')).strip().lower()
                result['translated_text'] = str(obj.get('translated_text', '')).strip()
                # Extract and validate brand - reject single characters and common errors
                raw_brand = str(obj.get('brand', '')).strip().upper()
                # Reject single-character brands (likely AI errors like 'm' instead of 'MSA')
                # Also reject common non-brand values
                invalid_brands = {'', 'N/A', 'NA', 'NONE', 'UNKNOWN', '-', 'NULL'}
                if len(raw_brand) <= 1 or raw_brand in invalid_brands:
                    result['brand'] = ''
                else:
                    result['brand'] = raw_brand
                result['concept_noun'] = str(obj.get('concept_noun', '')).strip().lower()
                result['modifier'] = str(obj.get('modifier', '')).strip()
                result['category_type'] = str(obj.get('category_type', '')).strip().lower()
                result['is_product_or_service'] = str(obj.get('is_product_or_service', 'Product')).strip()
                result['service_object'] = str(obj.get('service_object', '')).strip().lower()
                result['ai_insight'] = str(obj.get('ai_insight', '')).strip()
                result['confidence'] = str(obj.get('confidence', 'low')).strip().lower()
                result['mapped_category'] = str(obj.get('mapped_category', '')).strip()
                result['proposed_category'] = str(obj.get('proposed_category', '')).strip()
                result['matching_reason'] = str(obj.get('matching_reason', '')).strip()
                result['selected_category'] = str(obj.get('selected_category', '')).strip()  # For hierarchical classification
                
                # Map attributes if present - dynamically extract all returned attributes
                if self.extract_attributes and 'attributes' in obj:
                    attrs = obj['attributes']
                    # Dynamically extract all attributes returned by AI
                    result['attributes'] = {}
                    for key, value in attrs.items():
                        result['attributes'][key] = str(value).strip() if value else ''
                
        except json.JSONDecodeError as e:
            print(f"[Parse Error] Invalid JSON: {e}")
        except Exception as e:
            print(f"[Parse Error] {e}")
        
        return result
    
    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'language': '',
            'translated_text': '',
            'brand': '',
            'concept_noun': '',
            'modifier': '',
            'category_type': '',
            'is_product_or_service': 'Product',
            'service_object': '',
            'ai_insight': '',
            'confidence': 'low',
            'mapped_category': '',
            'proposed_category': '',
            'matching_reason': '',
            'selected_category': '',  # For hierarchical classification
            'attributes': {
                'material': '',
                'size_dimension': '',
                'uom': '',
                'thread_type': '',
                'pressure_rating': '',
                'temperature_rating': '',
                'voltage': '',
                'power': '',
                'quantity_per_pack': '',
                'packaging': '',
                'model': '',
                'processor': '',
                'ram': '',
                'storage': '',
                'screen_size': '',
                'software': '',
                'service_type': '',
                'duration': '',
                'scope': '',
                'other': ''
            }
        }
