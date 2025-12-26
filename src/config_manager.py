"""
Configuration Manager
Handles multiple taxonomy/keyword configurations
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import shutil

class ConfigManager:
    """Manage multiple taxonomy and keyword configurations"""
    
    def __init__(self, config_dir=None):
        if config_dir is None:
            config_dir = Path(r"C:\AI Opp\Projects\_Config")
        
        self.config_dir = Path(config_dir)
        self.cache_file = self.config_dir / "config_cache.json"
        self.configs_dir = self.config_dir / "Configurations"
        self.configs_dir.mkdir(exist_ok=True)
        
        # Load or create cache
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load configuration cache"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        else:
            # Create default cache
            default_cache = {
                "active_config": "Master",
                "configs": {
                    "Master": {
                        "taxonomy_file": "taxonomy.xlsx",
                        "keywords_file": "keywords.xlsx",
                        "created": datetime.now().strftime('%Y-%m-%d'),
                        "last_used": datetime.now().strftime('%Y-%m-%d'),
                        "description": "Master taxonomy configuration"
                    }
                },
                "settings": {
                    "min_keyword_matches": 2,
                    "conflict_threshold": 2,
                    "conflict_ratio": 2.0
                },
                "history": []
            }
            self._save_cache(default_cache)
            return default_cache
    
    def _save_cache(self, cache=None):
        """Save configuration cache"""
        if cache is None:
            cache = self.cache
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    
    def get_active_config(self):
        """Get active configuration name"""
        return self.cache.get("active_config", "Master")
    
    def get_config_info(self, config_name=None):
        """Get configuration information"""
        if config_name is None:
            config_name = self.get_active_config()
        
        return self.cache["configs"].get(config_name, {})
    
    def list_configs(self):
        """List all available configurations"""
        return list(self.cache["configs"].keys())
    
    def set_active_config(self, config_name):
        """Set active configuration"""
        if config_name in self.cache["configs"]:
            self.cache["active_config"] = config_name
            self.cache["configs"][config_name]["last_used"] = datetime.now().strftime('%Y-%m-%d')
            self._save_cache()
            
            # Add to history
            self._add_history("config_switched", f"Switched to {config_name}")
            return True
        return False
    
    def create_config(self, config_name, description="", taxonomy_file=None, keywords_file=None):
        """Create new configuration"""
        if config_name in self.cache["configs"]:
            return False, "Configuration already exists"
        
        # Create config directory
        config_path = self.configs_dir / config_name
        config_path.mkdir(exist_ok=True)
        
        # Copy files if provided
        if taxonomy_file:
            shutil.copy(taxonomy_file, config_path / "taxonomy.xlsx")
        
        if keywords_file:
            shutil.copy(keywords_file, config_path / "keywords.xlsx")
        
        # Add to cache
        self.cache["configs"][config_name] = {
            "taxonomy_file": "taxonomy.xlsx" if taxonomy_file else "",
            "keywords_file": "keywords.xlsx" if keywords_file else "",
            "created": datetime.now().strftime('%Y-%m-%d'),
            "last_used": datetime.now().strftime('%Y-%m-%d'),
            "description": description,
            "categories_count": 0,
            "keywords_count": 0
        }
        
        self._save_cache()
        self._add_history("config_created", f"Created {config_name}")
        
        return True, "Configuration created successfully"
    
    def delete_config(self, config_name):
        """Delete configuration"""
        if config_name == "Master":
            return False, "Cannot delete Master configuration"
        
        if config_name not in self.cache["configs"]:
            return False, "Configuration not found"
        
        # Delete directory
        config_path = self.configs_dir / config_name
        if config_path.exists():
            shutil.rmtree(config_path)
        
        # Remove from cache
        del self.cache["configs"][config_name]
        
        # If active, switch to Master
        if self.cache["active_config"] == config_name:
            self.cache["active_config"] = "Master"
        
        self._save_cache()
        self._add_history("config_deleted", f"Deleted {config_name}")
        
        return True, "Configuration deleted successfully"
    
    def get_taxonomy_path(self, config_name=None):
        """Get path to taxonomy file"""
        if config_name is None:
            config_name = self.get_active_config()
        
        config_info = self.get_config_info(config_name)
        
        if config_name == "Master":
            return self.config_dir / config_info.get("taxonomy_file", "taxonomy.xlsx")
        else:
            return self.configs_dir / config_name / config_info.get("taxonomy_file", "taxonomy.xlsx")
    
    def get_keywords_path(self, config_name=None):
        """Get path to keywords file"""
        if config_name is None:
            config_name = self.get_active_config()
        
        config_info = self.get_config_info(config_name)
        
        if config_name == "Master":
            return self.config_dir / config_info.get("keywords_file", "keywords.xlsx")
        else:
            return self.configs_dir / config_name / config_info.get("keywords_file", "keywords.xlsx")
    
    def update_config_stats(self, config_name=None):
        """Update configuration statistics"""
        if config_name is None:
            config_name = self.get_active_config()
        
        try:
            # Count categories
            taxonomy_path = self.get_taxonomy_path(config_name)
            if taxonomy_path.exists():
                df_tax = pd.read_excel(taxonomy_path)
                categories_count = len(df_tax)
            else:
                categories_count = 0
            
            # Count keywords
            keywords_path = self.get_keywords_path(config_name)
            if keywords_path.exists():
                df_kw = pd.read_excel(keywords_path)
                keywords_count = len(df_kw)
            else:
                keywords_count = 0
            
            # Update cache
            self.cache["configs"][config_name]["categories_count"] = categories_count
            self.cache["configs"][config_name]["keywords_count"] = keywords_count
            self._save_cache()
            
            return True, categories_count, keywords_count
        except Exception as e:
            return False, 0, 0
    
    def get_settings(self):
        """Get current settings"""
        return self.cache.get("settings", {
            "min_keyword_matches": 2,
            "conflict_threshold": 2,
            "conflict_ratio": 2.0
        })
    
    def update_settings(self, **kwargs):
        """Update settings"""
        if "settings" not in self.cache:
            self.cache["settings"] = {}
        
        self.cache["settings"].update(kwargs)
        self._save_cache()
        self._add_history("settings_updated", f"Updated settings: {kwargs}")
    
    def _add_history(self, action, description):
        """Add entry to history"""
        if "history" not in self.cache:
            self.cache["history"] = []
        
        self.cache["history"].append({
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "action": action,
            "description": description
        })
        
        # Keep only last 100 entries
        self.cache["history"] = self.cache["history"][-100:]
        self._save_cache()
    
    def get_history(self, limit=20):
        """Get recent history"""
        history = self.cache.get("history", [])
        return history[-limit:]
    
    def analyze_keyword_distribution(self, config_name=None):
        """Analyze keyword distribution across categories"""
        if config_name is None:
            config_name = self.get_active_config()
        
        try:
            taxonomy_path = self.get_taxonomy_path(config_name)
            keywords_path = self.get_keywords_path(config_name)
            
            if not taxonomy_path.exists() or not keywords_path.exists():
                return None
            
            df_tax = pd.read_excel(taxonomy_path)
            df_kw = pd.read_excel(keywords_path)
            
            # Assuming keywords file has 'Category' and 'Keyword' columns
            distribution = df_kw.groupby('Category').size().to_dict()
            
            # Calculate targets (simple version)
            results = []
            for _, row in df_tax.iterrows():
                category = row.get('Category', row.iloc[0])
                current_count = distribution.get(category, 0)
                target = max(10, int(15))  # Simplified target
                
                results.append({
                    'category': category,
                    'current': current_count,
                    'target': target,
                    'status': 'ok' if current_count >= target * 0.8 else 'low'
                })
            
            return results
        except Exception as e:
            return None