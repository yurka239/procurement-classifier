# Procurement Classifier v2.1

AI-powered procurement text classification tool with structured attribute extraction and normalization.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4%2F5-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **AI Classification** - GPT-5/GPT-4 powered product categorization
- **Hierarchical Attribute Selection** - Choose attributes by category with group controls
- **Custom Attributes** - Define your own attributes with descriptions and examples
- **47 Predefined Attributes** - Organized into 7 categories (Industrial, Electrical, IT, etc.)
- **Normalization Engine** - Fingerprinting & clustering for consistency
- **Taxonomy Mapping** - Match to your custom category hierarchy
- **Web Search Fallback** - Google search for low-confidence items
- **Cost Tracking** - Monitor API usage and costs
- **Batch Processing** - Process thousands of items efficiently

---

## Installation Guide (Step-by-Step)

### Step 1: Download This Project

1. Click the green **"Code"** button above
2. Click **"Download ZIP"**
3. Extract the ZIP to a folder on your computer (e.g., `C:\ProcurementClassifier`)

### Step 2: Install Python (if not already installed)

**Option A: Standard Python (Recommended for beginners)**
1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download Python 3.11 or later
3. Run the installer
4. ⚠️ **IMPORTANT:** Check the box **"Add Python to PATH"** at the bottom!
5. Click "Install Now"

**Option B: Anaconda (If you already use it)**
1. Go to [anaconda.com/download](https://www.anaconda.com/download)
2. Download and install Anaconda
3. Python is included automatically

### Step 3: Install Required Libraries

1. Open the project folder
2. **Double-click** `INSTALL.bat`
3. Wait for installation to complete (1-2 minutes)
4. You should see "Installation complete!" message

**If INSTALL.bat doesn't work:**
1. Open Command Prompt (search "cmd" in Windows)
2. Navigate to project folder: `cd C:\ProcurementClassifier`
3. Run: `pip install -r requirements.txt`

### Step 4: Get Your OpenAI API Key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Click **"Create new secret key"**
4. Copy the key (starts with `sk-proj-...`)
5. ⚠️ Save it somewhere safe - you can only see it once!

### Step 5: Configure the Tool

1. Open the `_Config` folder
2. Copy `config.ini.example` and rename it to `config.ini`
3. Open `config.ini` with Notepad
4. Paste your OpenAI API key:

```ini
[API_Keys]
openai_api_key = sk-proj-paste-your-key-here
```

5. Save and close

### Step 6: Run the Application

1. **Double-click** `RUN_APP.bat`
2. A command window will open
3. Your web browser will automatically open to `http://localhost:8501`
4. The tool is ready to use!

**To stop:** Close the command window or press `Ctrl+C`

---

## Tools Included

| Tool | Command | Description |
|------|---------|-------------|
| **Classifier** | `RUN_APP.bat` | Main classification with AI |
| **Clustering** | `RUN_CLUSTERING.bat` | Data cleaning & normalization |

---

## Input Format

Excel file (`.xlsx`) with your data:

| Description | Value | ... |
|-------------|-------|-----|
| BALL VALVE 2IN SS NPT | 150.00 | ... |
| MSA SAFETY GOGGLE HIGH TEMP | 45.00 | ... |

**Required:** `Description` column  
**Optional:** Value, Part Number, Manufacturer, etc.

---

## Output

| Column | Description |
|--------|-------------|
| Language | Detected language |
| Brand | Extracted brand (MSA, SKF, 3M, etc.) |
| Concept_Noun | Core product noun |
| Category | Matched taxonomy category |
| Manufacturer_Part_No | Part/model number |
| Material, Size, UOM... | Extracted attributes |

---

## 📋 Attribute Categories

The tool includes **47 predefined attributes** organized into 7 categories:

| Category | Icon | Attributes |
|----------|------|------------|
| **Industrial / MRO** | 🔧 | Material, Material_Grade, Size_Dimension, Thread_Type, Pressure_Rating, Temperature_Rating, Part_Model_Number, Connection_Type, Finish, Seal_Material |
| **Electrical** | ⚡ | Voltage, Power, Current, Phase, Frequency, IP_Rating, Electrical_Certification |
| **Packaging & Quantity** | 📦 | UOM, Quantity_Per_Pack, Packaging, Min_Order_Qty, Weight |
| **IT / Technology** | 💻 | Processor, RAM, Storage, Screen_Size, Resolution, Software, OS, Connectivity |
| **Services** | 🛠️ | Service_Type, Duration, Scope, SLA, Service_Frequency, Coverage_Area |
| **Safety / Construction** | 🏗️ | Color, Size_Apparel, Capacity, Safety_Standard, Hazard_Class, Protection_Level |
| **General** | 📝 | Country_Origin, Shelf_Life, Warranty, Certification, Other |

### Custom Attributes

You can create your own attributes with:
- **Name** - Attribute identifier (e.g., `Certification_Type`)
- **Description** - Explains to AI what to extract
- **Examples** - Sample values for AI guidance

---

## Project Structure

```
├── app.py                      # Main classifier app
├── clustering_app_v4_complete.py  # Clustering tool
├── RUN_APP.bat                 # Run classifier
├── RUN_CLUSTERING.bat          # Run clustering
├── INSTALL.bat                 # Install dependencies
├── requirements.txt            # Python packages
├── _Config/
│   └── config.ini.example      # Configuration template
├── _Templates/
│   ├── Taxonomy.xlsx           # Sample taxonomy
│   └── text_test.xlsx          # Sample data
└── src/                        # Core modules
    ├── ai_handler.py           # AI API integration
    ├── classifier_engine.py    # Classification logic
    ├── normalizer.py           # Fingerprinting engine
    └── ...
```

---

## Configuration

Edit `_Config/config.ini`:

| Setting | Default | Description |
|---------|---------|-------------|
| `default_model` | gpt-5-mini | AI model to use |
| `extract_attributes` | True | Extract structured attributes |
| `enable_normalization` | True | Apply fingerprinting |
| `use_web_search` | True | Search web for low-confidence |
| `max_workers` | 10 | Parallel API calls |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Install Python, check "Add to PATH" |
| "Module not found" | Run `INSTALL.bat` again |
| "Invalid API key" | Check key in `_Config/config.ini` |
| App won't start | Run `streamlit run app.py` manually |

---

## License

MIT License - See LICENSE file

---

**Version:** 2.1  
**Updated:** December 2024