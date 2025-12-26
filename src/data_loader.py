import pandas as pd

def load_taxonomy(filepath):
    """
    Load categories from taxonomy.xlsx
    
    Supports two formats:
    1. Single column: 'Category' (e.g., "MRO|MRO Parts")
    2. Separate columns: 'Level1' and 'Level2' (will be combined with |)
    """
    df = pd.read_excel(filepath, sheet_name=0)
    
    # Check for hierarchical format (Level1 + Level2 columns)
    if 'Level1' in df.columns and 'Level2' in df.columns:
        print(f"[Taxonomy] Detected hierarchical format (Level1|Level2)")
        # Combine Level1 and Level2 with pipe separator
        df['Category'] = df['Level1'].astype(str).str.strip() + '|' + df['Level2'].astype(str).str.strip()
        categories = df['Category'].dropna().tolist()
        categories = [c for c in categories if c and 'nan' not in c.lower()]
        
        # Show summary
        level1_count = df['Level1'].nunique()
        print(f"[Taxonomy] Loaded {len(categories)} categories ({level1_count} Level 1 categories)")
        return categories
    
    # Check for flat format (single Category column)
    elif 'Category' in df.columns:
        print(f"[Taxonomy] Detected flat format (Category column)")
        categories = df['Category'].dropna().astype(str).str.strip().tolist()
        categories = [c for c in categories if c]
        print(f"[Taxonomy] Loaded {len(categories)} categories")
        return categories
    
    else:
        raise ValueError("taxonomy.xlsx must have either 'Category' column OR 'Level1' and 'Level2' columns")

def load_input_data(filepath):
    """Load input data from Excel"""
    df = pd.read_excel(filepath, sheet_name=0)
    print(f"[Input Data] Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {', '.join(df.columns)}")
    return df