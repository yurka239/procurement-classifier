class CostTracker:
    """Track API usage and costs for AI and web search"""
    
    def __init__(self, price_table, model_name):
        self.price_table = price_table
        self.model_name = model_name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.rows_done = 0
        self.total_time = 0.0
        
        # Web search tracking
        self.google_searches = 0
        self.perplexity_searches = 0
    
    def add_usage(self, usage):
        """Track AI token usage (supports both object attributes and dict keys)"""
        if not usage:
            return
        try:
            # Handle dictionary format (from hierarchical classification)
            if isinstance(usage, dict):
                input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0) or 0
                output_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0) or 0
            # Handle object format (from standard classification)
            else:
                input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
            
            self.prompt_tokens += int(input_tokens)
            self.completion_tokens += int(output_tokens)
        except Exception as e:
            print(f"[Cost Tracker Warning] Failed to parse usage: {e}")
            pass
    
    def add_timing(self, secs):
        """Track processing time"""
        self.rows_done += 1
        self.total_time += secs
    
    def add_web_search(self, provider):
        """Track web search API calls"""
        if provider == 'google':
            self.google_searches += 1
        elif provider == 'perplexity':
            self.perplexity_searches += 1
    
    @property
    def avg_time_per_row(self):
        return self.total_time / self.rows_done if self.rows_done else 0.0
    
    def ai_cost(self):
        """Calculate AI model costs only"""
        if self.model_name not in self.price_table:
            return 0.0
        p = self.price_table[self.model_name]
        return (self.prompt_tokens / 1_000_000) * p['input'] + (self.completion_tokens / 1_000_000) * p['output']
    
    def web_search_cost(self):
        """Calculate web search API costs"""
        # Google Custom Search API: $5 per 1000 queries (first 100/day free)
        google_cost = (self.google_searches / 1000) * 5.0
        
        # Perplexity API: Token-based pricing (estimate)
        # sonar: ~$0.001-0.002 per search
        # sonar-pro: ~$0.003-0.005 per search
        # Using conservative estimate of $0.003/search
        perplexity_cost = self.perplexity_searches * 0.003
        
        return google_cost + perplexity_cost
    
    def total_cost(self):
        """Calculate total cost (AI + web search)"""
        return self.ai_cost() + self.web_search_cost()
    
    # Backward compatibility
    def cost_estimate(self):
        """Alias for total_cost for backward compatibility"""
        return self.total_cost()
    
    @staticmethod
    def estimate_cost(num_rows, keyword_match_rate, web_search_rate, model_name, price_table, web_provider='perplexity'):
        """Estimate cost before running classification
        
        Args:
            num_rows: Number of products to classify
            keyword_match_rate: % of products with strong keyword matches (0.0-1.0)
            web_search_rate: % of products needing web search (0.0-1.0)
            model_name: AI model to use
            price_table: Pricing table from config
            web_provider: 'google' or 'perplexity'
        
        Returns:
            dict with cost breakdown and time estimate
        """
        # AI calls: products without strong keyword matches
        ai_calls = int(num_rows * (1 - keyword_match_rate))
        
        # Estimate tokens per AI call
        # Step 2: ~800 tokens input, ~100 tokens output
        # Step 3 (with web): ~1500 tokens input, ~150 tokens output
        web_searches = int(num_rows * web_search_rate)
        regular_ai = ai_calls - web_searches
        
        # Token estimates
        input_tokens = (regular_ai * 800) + (web_searches * 1500)
        output_tokens = (regular_ai * 100) + (web_searches * 150)
        
        # AI cost
        if model_name in price_table:
            p = price_table[model_name]
            ai_cost = (input_tokens / 1_000_000) * p['input'] + (output_tokens / 1_000_000) * p['output']
        else:
            ai_cost = 0.0
        
        # Web search cost
        if web_provider == 'google':
            web_cost = (web_searches / 1000) * 5.0
        else:  # perplexity
            web_cost = web_searches * 0.003
        
        total_cost = ai_cost + web_cost
        
        # Time estimate (parallel processing with 10 workers)
        # Keyword matching: ~0.1s per product
        # AI classification: ~8s per product (parallelized)
        # Web search: +15s per product with web search
        keyword_time = num_rows * 0.1
        ai_time = (regular_ai * 8) / 10  # Parallel with 10 workers
        web_time = (web_searches * 15) / 10  # Parallel
        total_time = keyword_time + ai_time + web_time
        
        return {
            'total_rows': num_rows,
            'keyword_matches': int(num_rows * keyword_match_rate),
            'ai_calls': ai_calls,
            'web_searches': web_searches,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'ai_cost': ai_cost,
            'web_cost': web_cost,
            'total_cost': total_cost,
            'estimated_time_sec': total_time,
            'estimated_time_min': total_time / 60
        }
    
    def get_summary(self):
        """Return cost summary as dictionary"""
        return {
            'model_name': self.model_name,
            'total_requests': self.rows_done,
            'total_input_tokens': self.prompt_tokens,
            'total_output_tokens': self.completion_tokens,
            'total_tokens': self.prompt_tokens + self.completion_tokens,
            'google_searches': self.google_searches,
            'perplexity_searches': self.perplexity_searches,
            'total_searches': self.google_searches + self.perplexity_searches,
            'ai_cost': self.ai_cost(),
            'web_search_cost': self.web_search_cost(),
            'total_cost': self.total_cost(),
            'total_time': self.total_time,
            'avg_time_per_row': self.avg_time_per_row
        }
    
    def print_summary(self):
        """Print detailed cost and performance summary"""
        print("\n" + "="*60)
        print("COST & PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Rows processed: {self.rows_done}")
        print(f"Total time: {self.total_time:.1f}s ({self.total_time/60:.1f} min)")
        print(f"Avg time/row: {self.avg_time_per_row:.2f}s")
        
        print("AI Tokens:")
        print(f"  Input:  {self.prompt_tokens:,}")
        print(f"  Output: {self.completion_tokens:,}")
        print(f"  Total:  {self.prompt_tokens + self.completion_tokens:,}")
        
        print("Web Search:")
        print(f"  Google searches: {self.google_searches}")
        print(f"  Perplexity searches: {self.perplexity_searches}")
        print(f"  Total searches: {self.google_searches + self.perplexity_searches}")
        
        print("Cost Breakdown:")
        print(f"  AI model: ${self.ai_cost():.4f}")
        print(f"  Web search: ${self.web_search_cost():.4f}")
        print(f"  Total: ${self.total_cost():.4f}")
        print("="*60)
