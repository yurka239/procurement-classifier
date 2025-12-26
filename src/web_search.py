import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def search_google(query, api_key, cse_id, num_results=3):
    """Search Google for product information with URLs"""
    if not api_key or not cse_id or not query:
        print("[Google Search] Missing required parameters")
        if not api_key:
            print("  - API key is missing")
        if not cse_id:
            print("  - CSE ID is missing")
        if not query:
            print("  - Query is empty")
        return []
    
    try:
        service = build("customsearch", "v1", developerKey=api_key, cache_discovery=False)
        res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        items = res.get("items", []) or []
        
        # Return results as list of dicts with snippet and URL separated
        results = []
        for item in items:
            snippet = item.get("snippet", "")
            url = item.get("link", "")
            if snippet:
                # Return dict so we can access snippet and URL separately
                results.append({"snippet": snippet, "url": url})
        
        if results:
            print(f"[Google Search] Found {len(results)} results for: {query[:50]}...")
        else:
            print(f"[Google Search] No results found for: {query[:50]}...")
        
        return results
        
    except HttpError as e:
        error_details = e.error_details if hasattr(e, 'error_details') else str(e)
        print(f"[Google Search] HTTP Error: {error_details}")
        if "quotaExceeded" in str(e):
            print("  ⚠️ Daily quota exceeded. Check your Google Cloud Console.")
        elif "invalid" in str(e).lower():
            print("  ⚠️ Invalid API key or CSE ID. Please verify your credentials.")
        return []
        
    except Exception as e:
        print(f"[Google Search] Unexpected error: {type(e).__name__}: {str(e)}")
        return []

def search_perplexity(query, api_key, categories=None, model="sonar"):
    """Search using Perplexity AI"""
    if not api_key or not query:
        return []
    
    try:
        url = "https://api.perplexity.ai/chat/completions"
        
        system_prompt = """You are an expert in industrial and commercial product classification. 
IMPORTANT: Only provide information if you have reliable data about the product.
If you cannot find specific information, respond with: "INSUFFICIENT DATA - Unable to classify this product."
Do not guess or infer based solely on manufacturer name."""
        
        if categories and len(categories) > 0:
            cat_sample = ', '.join(categories[:20])
            if len(categories) > 20:
                cat_sample += f"... and {len(categories)-20} more"
            
            user_prompt = f"""Analyze this product:

Product: {query}

Available Categories: {cat_sample}

Provide:
1. Product type and function
2. Industry/application
3. Key specifications
4. Best matching category"""
        else:
            user_prompt = f"""Analyze this product:

Product: {query}

Provide product type, use case, and technical specifications."""
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Try with 90s timeout, retry once if timeout
        max_retries = 2
        for attempt in range(max_retries):
            try:
                timeout = 90 if attempt == 0 else 120  # Increase timeout on retry
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                break  # Success, exit retry loop
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"[Perplexity] Timeout, retrying... (attempt {attempt+2}/{max_retries})")
                    continue
                else:
                    print(f"[Perplexity] Timeout after {max_retries} attempts")
                    return []
        
        response_data = response.json()
        choice = response_data.get('choices', [{}])[0]
        content = choice.get('message', {}).get('content', '')
        
        # Extract citations (URLs) if available
        citations = response_data.get('citations', [])
        
        # Return content with citations as dict (similar to Google format)
        if content:
            result = {
                'snippet': content,
                'url': citations[0] if citations else 'Perplexity AI',
                'citations': citations  # All citations for reference
            }
            return [result]
        return []
        
    except Exception as e:
        return []
