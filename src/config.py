import os
import re
from pathlib import Path
import configparser
import requests

class Config:
    """Load and manage configuration from config.ini"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Auto-detect config.ini in _Config folder relative to this script
            script_dir = Path(__file__).parent
            config_path = script_dir.parent / '_Config' / 'config.ini'
        
        # Ensure config_path is a Path object
        config_path = Path(config_path)
        
        # Check if config file exists - if not, try to create from example
        if not config_path.exists():
            example_path = config_path.parent / 'config.ini.example'
            if example_path.exists():
                # Copy example to config.ini for first-time setup
                import shutil
                shutil.copy(example_path, config_path)
                print(f"[Config] Created config.ini from example. Please add your API keys!")
                print(f"[Config] Edit: {config_path}")
            else:
                raise FileNotFoundError(
                    f"\n" + "="*60 + f"\n"
                    f"FIRST TIME SETUP REQUIRED\n"
                    f"="*60 + f"\n\n"
                    f"1. Go to: {config_path.parent}\n"
                    f"2. Copy 'config.ini.example' to 'config.ini'\n"
                    f"3. Open 'config.ini' and add your OpenAI API key\n"
                    f"4. Run the app again\n\n"
                    f"Get your API key at: https://platform.openai.com/api-keys\n"
                    f"="*60
                )
        
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')
        
        # Paths
        projects_dir_str = self.config['Paths'].get('projects_directory', '')
        if projects_dir_str and projects_dir_str.strip():
            self.projects_dir = Path(projects_dir_str)
        else:
            # Auto-detect: create 'Output' folder in app directory
            app_root = Path(config_path).parent.parent if config_path else Path.cwd()
            self.projects_dir = app_root / 'Output'
            self.projects_dir.mkdir(exist_ok=True)
        
        taxonomy_file_str = self.config['Paths'].get('taxonomy_file', '')
        self.taxonomy_file = Path(taxonomy_file_str) if taxonomy_file_str else Path()
        
        # Legacy file-based keys (for backward compatibility)
        self.openai_key_file = Path(self.config['Paths'].get('openai_key_file', '')) if 'openai_key_file' in self.config['Paths'] else None
        self.google_key_file = Path(self.config['Paths'].get('google_key_file', '')) if 'google_key_file' in self.config['Paths'] else None
        self.google_cse_id_file = Path(self.config['Paths'].get('google_cse_id_file', '')) if 'google_cse_id_file' in self.config['Paths'] else None
        self.perplexity_key_file = Path(self.config['Paths'].get('perplexity_key_file', '')) if 'perplexity_key_file' in self.config['Paths'] else None
        
        # Model settings
        self.default_model = self.config['Model']['default_model']
        self.available_models = [m.strip() for m in self.config['Model']['available_models'].split(',')]
        self.gpt5_verbosity = self.config['Model'].get('gpt5_verbosity', 'low')
        
        # Build pricing table
        self.price_table = {}
        for model in self.available_models:
            model_key = model.replace('-', '_')
            if f'{model_key}_input_price' in self.config['Model']:
                self.price_table[model] = {
                    'input': float(self.config['Model'][f'{model_key}_input_price']),
                    'output': float(self.config['Model'][f'{model_key}_output_price'])
                }
        
        # Classification settings
        self.taxonomy_fuzzy_threshold = float(self.config['Classification'].get('taxonomy_fuzzy_threshold', '0.9'))
        self.use_web_search = self.config['Classification'].getboolean('use_web_search')
        self.min_confidence_for_web = self.config['Classification'].get('min_confidence_for_web', 'medium')
        self.web_search_provider = self.config['Classification']['web_search_provider']
        self.web_search_results = int(self.config['Classification'].get('web_search_results', '3'))
        self.perplexity_model = self.config['Classification'].get('perplexity_model', 'sonar')
        self.web_search_confidence_levels = self.config['Classification'].get('web_search_confidence_levels', 'medium,low,blank')
        
        # Performance settings
        self.save_checkpoint_every = int(self.config['Performance']['save_checkpoint_every'])
        self.show_progress_every = int(self.config['Performance']['show_progress_every'])
        self.max_workers = int(self.config['Performance'].get('max_workers', '10'))  # Parallel processing workers
        
        # Load API keys (priority: env vars > streamlit secrets > config.ini > files)
        self.openai_key = self._load_api_key('API_Keys', 'openai_api_key', self.openai_key_file, env_var='OPENAI_API_KEY')
        
        # Debug: show what key was loaded
        if self.openai_key:
            print(f"[Config] OpenAI key loaded, starts with: {self.openai_key[:12]}...")
        else:
            print("[Config] WARNING: No OpenAI key found!")
        
        # Validate OpenAI key (check for placeholder text)
        # Accept any key starting with 'sk-' (case-insensitive: sk-, Sk-, SK-)
        if not self.openai_key or 'your-' in self.openai_key.lower() or 'paste' in self.openai_key.lower() or not self.openai_key.lower().startswith('sk-'):
            # For cloud deployment, provide a clearer message
            try:
                import streamlit as st
                is_cloud = hasattr(st, 'secrets')
            except:
                is_cloud = False
            
            if is_cloud:
                raise ValueError(
                    "OPENAI API KEY REQUIRED\n\n"
                    "Add 'openai_api_key' to your Streamlit Secrets:\n"
                    "  openai_api_key = \"sk-...\"\n\n"
                    "(Accepts sk-proj, sk-svcacct, or other valid OpenAI keys)\n\n"
                    "Go to: App Settings → Secrets"
                )
            else:
                raise ValueError(
                    f"\n" + "="*60 + f"\n"
                    f"OPENAI API KEY REQUIRED\n"
                    f"="*60 + f"\n\n"
                    f"Please add your OpenAI API key to:\n"
                    f"  {config_path}\n\n"
                    f"Open the file and replace the placeholder with your key:\n"
                    f"  openai_api_key = sk-proj-YOUR-ACTUAL-KEY\n\n"
                    f"Get your API key at: https://platform.openai.com/api-keys\n"
                    f"="*60
                )
        
        if self.web_search_provider == 'google':
            try:
                self.google_api_key = self._load_api_key('API_Keys', 'google_api_key', self.google_key_file, validator=self._extract_google_key, env_var='GOOGLE_API_KEY')
                self.google_cse_id = self._load_api_key('API_Keys', 'google_cse_id', self.google_cse_id_file, validator=self._extract_cse_id, env_var='GOOGLE_CSE_ID')
                
                if self.google_api_key and self.google_cse_id:
                    print(f"[Config] Google API key loaded: {self.google_api_key[:10]}...")
                    print(f"[Config] Google CSE ID loaded: {self.google_cse_id}")
                else:
                    print("[Config] Google keys not configured")
                
                self.perplexity_key = None
            except Exception as e:
                print(f"[Config] ERROR loading Google keys: {e}")
                self.google_api_key = None
                self.google_cse_id = None
                self.perplexity_key = None
        elif self.web_search_provider == 'perplexity':
            self.perplexity_key = self._load_api_key('API_Keys', 'perplexity_api_key', self.perplexity_key_file, env_var='PERPLEXITY_API_KEY')
            self.google_api_key = None
            self.google_cse_id = None
        else:
            self.google_api_key = None
            self.google_cse_id = None
            self.perplexity_key = None
        
        print("[Config] Loaded successfully")
        print(f"  • Default model: {self.default_model}")
        print(f"  • Available models: {', '.join(self.available_models)}")
        print(f"  • Web search: {self.web_search_provider if self.use_web_search else 'Disabled'}")
        if self.web_search_provider == 'perplexity':
            print(f"  • Perplexity model: {self.perplexity_model}")
            print(f"  • Enhancement triggers: {self.web_search_confidence_levels}")
    
    def _load_api_key(self, section, key, file_path=None, validator=None, env_var=None):
        """
        Load API key with priority: Environment Variable > config.ini > file path
        This allows cloud deployment (env vars) while keeping local dev (config.ini) working.
        """
        # 1. Try environment variable first (for cloud deployment)
        if env_var:
            env_value = os.environ.get(env_var, '').strip()
            if env_value:
                return env_value
        
        # 2. Try Streamlit secrets (for Streamlit Cloud)
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                # Try lowercase key (e.g., openai_api_key)
                if key in st.secrets:
                    val = str(st.secrets[key]).strip()
                    if val:
                        print(f"[Config] Loaded {key} from Streamlit secrets")
                        return val
                # Try uppercase env var name (e.g., OPENAI_API_KEY)
                if env_var and env_var in st.secrets:
                    val = str(st.secrets[env_var]).strip()
                    if val:
                        print(f"[Config] Loaded {env_var} from Streamlit secrets")
                        return val
        except Exception as e:
            print(f"[Config] Error reading Streamlit secrets: {e}")
        
        # 3. Try direct key in config.ini (for local development)
        direct_key = self.config[section].get(key, '').strip()
        if direct_key:
            return direct_key
        
        # 4. Fall back to file-based key (backward compatibility)
        if file_path and file_path.exists():
            if validator:
                return validator(file_path)
            else:
                return self._read_file(file_path).strip()
        
        return None
    
    @staticmethod
    def _extract_google_key(path):
        """Extract and validate Google API key from file"""
        raw = Config._read_file(path)
        m = re.search(r"AIza[0-9A-Za-z_\-]+", raw)
        if not m:
            raise ValueError("Google API key must start with 'AIza'")
        return m.group(0)
    
    @staticmethod
    def _extract_cse_id(path):
        """Extract and validate Google CSE ID from file"""
        raw = Config._read_file(path).strip()
        if re.fullmatch(r"[0-9A-Za-z]{17}", raw):
            return raw
        m = re.search(r"(?:[?&])cx=([0-9A-Za-z]{17})", raw)
        if m:
            return m.group(1)
        raise ValueError("Invalid Google CSE ID format")
    
    @staticmethod
    def _read_file(path):
        return open(path, 'r', encoding='utf-8').read()
    
    @staticmethod
    def _read_google_key(path):
        raw = Config._read_file(path)
        m = re.search(r"AIza[0-9A-Za-z_\-]+", raw)
        if not m:
            raise ValueError("Google API key must start with 'AIza'")
        return m.group(0)
    
    @staticmethod
    def _read_cse_id(path):
        raw = Config._read_file(path).strip()
        if re.fullmatch(r"[0-9A-Za-z]{17}", raw):
            return raw
        m = re.search(r"(?:[?&])cx=([0-9A-Za-z]{17})", raw)
        if m:
            return m.group(1)
        raise ValueError("Invalid Google CSE ID format")
    
    @staticmethod
    def check_google_keys(api_key, cse_id, test_query="test"):
        """
        Verifies that Google API key and CSE ID work with Custom Search API.
        Returns (success: bool, message: str)
        """
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {"key": api_key, "cx": cse_id, "q": test_query, "num": 1}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if response.status_code == 200 and "items" in data:
                return True, f"✅ Google API valid (found {len(data['items'])} results)"
            elif "error" in data:
                err_msg = data["error"].get("message", "Unknown error")
                err_code = data["error"].get("code", "")
                return False, f"❌ Google API error [{err_code}]: {err_msg}"
            else:
                return False, f"⚠️ Unexpected response: {data}"
        except requests.exceptions.Timeout:
            return False, "❌ Google API timeout (check internet connection)"
        except Exception as e:
            return False, f"❌ Error checking Google keys: {str(e)}"
    
    @staticmethod
    def check_openai_key(api_key):
        """
        Verifies OpenAI API key by making a minimal API call.
        Returns (success: bool, message: str)
        """
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            # Make minimal call to verify key
            response = client.models.list()
            return True, "✅ OpenAI API key valid"
        except openai.AuthenticationError:
            return False, "❌ OpenAI API key invalid or expired"
        except Exception as e:
            return False, f"❌ Error checking OpenAI key: {str(e)}"
    
    @staticmethod
    def check_perplexity_key(api_key):
        """
        Verifies Perplexity API key by making a minimal API call.
        Returns (success: bool, message: str)
        """
        try:
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "sonar",
                "messages": [{"role": "user", "content": "test"}]
            }
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return True, "✅ Perplexity API key valid"
            elif response.status_code == 401:
                return False, "❌ Perplexity API key invalid or expired"
            else:
                return False, f"❌ Perplexity API error: {response.status_code}"
        except requests.exceptions.Timeout:
            return False, "❌ Perplexity API timeout (check internet connection)"
        except Exception as e:
            return False, f"❌ Error checking Perplexity key: {str(e)}"
    
    def validate_api_keys(self, verbose=True):
        """
        Validate all configured API keys.
        Returns dict with validation results.
        """
        results = {}
        
        # Check OpenAI key
        if self.openai_key:
            success, msg = self.check_openai_key(self.openai_key)
            results['openai'] = {'valid': success, 'message': msg}
            if verbose:
                print(f"  {msg}")
        
        # Check Google keys if configured
        if self.web_search_provider == 'google' and self.google_api_key and self.google_cse_id:
            success, msg = self.check_google_keys(self.google_api_key, self.google_cse_id)
            results['google'] = {'valid': success, 'message': msg}
            if verbose:
                print(f"  {msg}")
        
        # Check Perplexity key if configured
        if self.web_search_provider == 'perplexity' and self.perplexity_key:
            success, msg = self.check_perplexity_key(self.perplexity_key)
            results['perplexity'] = {'valid': success, 'message': msg}
            if verbose:
                print(f"  {msg}")
        
        return results
    
    def load_attribute_config(self, taxonomy_file, sets_sheet='Attribute_Sets', mapping_sheet='Set_Mapping'):
        """
        Load attribute sets and category mappings from Excel file.
        
        Args:
            taxonomy_file: Path to Excel file containing attribute configuration
            sets_sheet: Name of sheet containing attribute set definitions
            mapping_sheet: Name of sheet containing category-to-set mappings
            
        Returns:
            tuple: (attribute_sets dict, category_to_set dict)
        """
        import pandas as pd
        
        try:
            # Load attribute sets
            sets_df = pd.read_excel(taxonomy_file, sheet_name=sets_sheet)
            attribute_sets = {}
            
            for _, row in sets_df.iterrows():
                set_name = row['Attribute_Set']
                attrs = []
                for i in range(1, 9):  # Att1 through Att8
                    col_name = f'Att{i}'
                    if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip():
                        attrs.append(str(row[col_name]).strip())
                
                if attrs:  # Only add if there are attributes
                    attribute_sets[set_name] = attrs
            
            # Load category mappings
            mapping_df = pd.read_excel(taxonomy_file, sheet_name=mapping_sheet)
            category_to_set = {}
            
            for _, row in mapping_df.iterrows():
                category = str(row['Level1_Category']).strip()
                attr_set = str(row['Attribute_Set']).strip()
                category_to_set[category] = attr_set
            
            print(f"[Config] Loaded {len(attribute_sets)} attribute sets and {len(category_to_set)} category mappings")
            return attribute_sets, category_to_set
            
        except Exception as e:
            print(f"[Config] Warning: Could not load attribute configuration: {e}")
            print(f"[Config] Using default MRO_Set for all categories")
            
            # Return comprehensive default set covering most categories
            default_set = {
                'Default_Set': [
                    'Material', 'Size_Dimension', 'UOM', 'Thread_Type', 
                    'Pressure_Rating', 'Temperature_Rating', 'Voltage', 'Power',
                    'Quantity_Per_Pack', 'Packaging', 'Model', 'Processor',
                    'RAM', 'Storage', 'Screen_Size', 'Software', 'Service_Type',
                    'Duration', 'Scope', 'Other'
                ]
            }
            return default_set, {}
