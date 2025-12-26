"""
Utility functions for UI components
"""

import streamlit as st

def get_default_ai_prompt():
    """Returns the default AI classification prompt"""
    return """You are an expert product classification system.

Your task: Analyze the product and extract key information.

Product Information:
{product_info}

{categories}

Extract the following information:

1. **Compact_Noun**: The main item type as a short plural noun or noun phrase
   - Examples: "gloves", "bearings", "seals", "sprays", "helmets", "caps"
   - Always plural unless inherently singular (e.g., "service", "equipment")
   - Ignore materials, colors, dimensions, brands, quantities
   - For services: "service (category)" — e.g., "service (transport)"

2. **Category_Type**: Functional group inferred from context
   - Common patterns:
     * "seal", "packing", "O-ring", "gland", "ring" → sealing components
     * "bearing", "pillow block", "bushing", "sleeve" → bearing components
     * "piston", "cylinder" → piston components
     * PPE/safety gear → specify type (hand protection, head protection, etc.)
     * "service", "installation", "transport" → service (domain)
     * Otherwise → mechanical components

3. **AI_Insight**: Brief explanation of what the product is (10-15 words)

4. **AI_Confidence**: Your confidence level (high/medium/low)

Respond with JSON (all text in English):
{{
  "compact_noun": "<core plural item noun>",
  "category_type": "<functional group>",
  "ai_insight": "<brief product description>",
  "confidence": "high/medium/low",
  "category": "<exact match from valid categories or empty>",
  "proposed_category": "<suggested category if not in valid list>"
}}
"""

def get_default_web_prompt():
    """Returns the default web search prompt"""
    return """Based on the web search results, classify this product:

Product: {description}

Web Search Results:
{snippets}

Extract the same information as AI classification."""


def format_confidence(confidence):
    """Format confidence level with emoji"""
    conf_map = {
        'high': '🟢 High',
        'medium': '🟡 Medium',
        'low': '🔴 Low',
        'none': '⚫ None'
    }
    return conf_map.get(confidence.lower(), confidence)


def estimate_cost(num_rows, model_name, price_table):
    """Estimate classification cost and time"""
    if model_name not in price_table:
        return {
            'cost': 'Unknown',
            'cost_formatted': 'Unknown',
            'time_seconds': 0,
            'time_formatted': 'Unknown'
        }
    
    # Rough estimate: ~500 tokens input, ~150 tokens output per row
    input_tokens = num_rows * 500
    output_tokens = num_rows * 150
    
    input_cost = (input_tokens / 1_000_000) * price_table[model_name]['input']
    output_cost = (output_tokens / 1_000_000) * price_table[model_name]['output']
    total_cost = input_cost + output_cost
    
    # Time estimate: ~3-5 seconds per row (SEQUENTIAL processing)
    # GPT-5: ~3s, GPT-4: ~4s average
    # Note: Currently using sequential processing (one row at a time)
    time_per_row = 3.5 if model_name.startswith('gpt-5') else 4.0
    total_seconds = num_rows * time_per_row
    
    # Format time
    if total_seconds < 60:
        time_formatted = f"{total_seconds:.0f}s"
    elif total_seconds < 3600:
        time_formatted = f"{total_seconds/60:.1f} min"
    else:
        time_formatted = f"{total_seconds/3600:.1f} hours"
    
    return {
        'cost': total_cost,
        'cost_formatted': f"${total_cost:.2f}",
        'time_seconds': total_seconds,
        'time_formatted': time_formatted
    }