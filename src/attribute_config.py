"""Attribute Configuration for Procurement Classifier v2.1

Defines all available attributes organized by category.
Users can select from predefined attributes or add custom ones.
"""

from typing import Dict, List, Any

# Master attribute categories with all available attributes
ATTRIBUTE_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Industrial_MRO": {
        "icon": "🔧",
        "display_name": "Industrial / MRO",
        "description": "Mechanical, piping, and general MRO attributes",
        "attributes": {
            "Material": {
                "key": "material",
                "description": "Primary material composition",
                "examples": "stainless steel, brass, PVC, carbon steel, aluminum, copper"
            },
            "Material_Grade": {
                "key": "material_grade",
                "description": "Material specification or grade",
                "examples": "316, 304, A105, ASTM A36, Grade 8, 6061-T6"
            },
            "Size_Dimension": {
                "key": "size_dimension",
                "description": "Main dimension with unit",
                "examples": "2in, 50mm, DN50, 1/2\", 3/4 inch, 100x50mm"
            },
            "Thread_Type": {
                "key": "thread_type",
                "description": "Thread specification",
                "examples": "NPT, BSP, metric, UNC, UNF, M10x1.5, BSPP"
            },
            "Pressure_Rating": {
                "key": "pressure_rating",
                "description": "Pressure class or rating",
                "examples": "150 psi, PN16, 300#, Class 150, 3000 PSI, ANSI 150"
            },
            "Temperature_Rating": {
                "key": "temperature_rating",
                "description": "Operating temperature range",
                "examples": "-40 to 200°C, max 150°F, -20°C to 80°C, cryogenic"
            },
            "Part_Model_Number": {
                "key": "part_model_number",
                "description": "Product part number, model number, or catalog code",
                "examples": "SKF-6205-2RS, 10115859, ThinkPad T14, BV-200, 3M-2090"
            },
            "Connection_Type": {
                "key": "connection_type",
                "description": "How the item connects or attaches",
                "examples": "flanged, threaded, welded, socket weld, tri-clamp, compression"
            },
            "Finish": {
                "key": "finish",
                "description": "Surface finish or treatment",
                "examples": "galvanized, chrome plated, painted, anodized, polished, raw"
            },
            "Seal_Material": {
                "key": "seal_material",
                "description": "Seal or gasket material",
                "examples": "PTFE, Viton, EPDM, Buna-N, graphite, metal-to-metal"
            }
        }
    },
    
    "Electrical": {
        "icon": "⚡",
        "display_name": "Electrical",
        "description": "Electrical specifications and ratings",
        "attributes": {
            "Voltage": {
                "key": "voltage",
                "description": "Electrical voltage specification",
                "examples": "24V DC, 110V AC, 220V, 480V 3-phase, 12V"
            },
            "Power": {
                "key": "power",
                "description": "Power rating",
                "examples": "1.5 kW, 500W, 3 HP, 750 watts, 0.5 HP"
            },
            "Current": {
                "key": "current",
                "description": "Electrical current/amperage",
                "examples": "10A, 15 amps, 20mA, 4-20mA"
            },
            "Phase": {
                "key": "phase",
                "description": "Electrical phase configuration",
                "examples": "single phase, 3-phase, DC"
            },
            "Frequency": {
                "key": "frequency",
                "description": "Operating frequency",
                "examples": "50Hz, 60Hz, 50/60Hz"
            },
            "IP_Rating": {
                "key": "ip_rating",
                "description": "Ingress protection rating",
                "examples": "IP65, IP67, IP68, NEMA 4X, NEMA 12"
            },
            "Electrical_Certification": {
                "key": "electrical_certification",
                "description": "Electrical certifications",
                "examples": "UL, CE, ATEX, CSA, FM, IECEx"
            }
        }
    },
    
    "Packaging_Quantity": {
        "icon": "📦",
        "display_name": "Packaging & Quantity",
        "description": "Packaging and quantity specifications",
        "attributes": {
            "UOM": {
                "key": "uom",
                "description": "Unit of measure",
                "examples": "each, box, meter, kg, liter, pair, roll, foot"
            },
            "Quantity_Per_Pack": {
                "key": "quantity_per_pack",
                "description": "Quantity per package/box",
                "examples": "100, 50 pcs, box of 25, 1000/box, pack of 10"
            },
            "Packaging": {
                "key": "packaging",
                "description": "Packaging type",
                "examples": "box, pallet, drum, bag, roll, carton, bulk"
            },
            "Min_Order_Qty": {
                "key": "min_order_qty",
                "description": "Minimum order quantity",
                "examples": "10 pcs, 1 case, 100 minimum, MOQ 50"
            },
            "Weight": {
                "key": "weight",
                "description": "Item or package weight",
                "examples": "5 kg, 10 lbs, 500g, 2.5 pounds"
            }
        }
    },
    
    "IT_Technology": {
        "icon": "💻",
        "display_name": "IT / Technology",
        "description": "IT equipment and technology specifications",
        "attributes": {
            "Processor": {
                "key": "processor",
                "description": "CPU/processor type",
                "examples": "Intel i7, AMD Ryzen 5, Apple M1, Snapdragon 8"
            },
            "RAM": {
                "key": "ram",
                "description": "Memory specification",
                "examples": "16GB, 32GB DDR4, 8GB RAM, 64GB DDR5"
            },
            "Storage": {
                "key": "storage",
                "description": "Storage capacity",
                "examples": "512GB SSD, 1TB HDD, 256GB NVMe, 2TB"
            },
            "Screen_Size": {
                "key": "screen_size",
                "description": "Display size",
                "examples": "15.6 inch, 27\", 24 inch, 14\" FHD"
            },
            "Resolution": {
                "key": "resolution",
                "description": "Display resolution",
                "examples": "1920x1080, 4K, 2560x1440, Full HD, Retina"
            },
            "Software": {
                "key": "software",
                "description": "Software name or type",
                "examples": "Microsoft Office, AutoCAD, Windows 11, Adobe CC"
            },
            "OS": {
                "key": "os",
                "description": "Operating system",
                "examples": "Windows 11, macOS, Linux, iOS, Android"
            },
            "Connectivity": {
                "key": "connectivity",
                "description": "Connection/interface types",
                "examples": "WiFi 6, Bluetooth 5.0, USB-C, Thunderbolt 4, HDMI"
            }
        }
    },
    
    "Services": {
        "icon": "🛠️",
        "display_name": "Services",
        "description": "Service-related specifications",
        "attributes": {
            "Service_Type": {
                "key": "service_type",
                "description": "Type of service",
                "examples": "maintenance, repair, installation, consulting, calibration"
            },
            "Duration": {
                "key": "duration",
                "description": "Service duration or contract period",
                "examples": "1 year, 6 months, annual, quarterly, 3-year contract"
            },
            "Scope": {
                "key": "scope",
                "description": "Service scope or coverage",
                "examples": "on-site, remote, 24/7 support, preventive maintenance"
            },
            "SLA": {
                "key": "sla",
                "description": "Service level agreement",
                "examples": "4-hour response, next-day, 99.9% uptime, 24/7 support"
            },
            "Service_Frequency": {
                "key": "service_frequency",
                "description": "Service frequency",
                "examples": "weekly, monthly, quarterly, annual, on-demand"
            },
            "Coverage_Area": {
                "key": "coverage_area",
                "description": "Geographic coverage",
                "examples": "nationwide, regional, on-site only, global"
            }
        }
    },
    
    "Safety_Construction": {
        "icon": "🏗️",
        "display_name": "Safety / Construction",
        "description": "Safety equipment and construction specifications",
        "attributes": {
            "Color": {
                "key": "color",
                "description": "Color or finish",
                "examples": "red, blue, safety orange, high-vis yellow, black"
            },
            "Size_Apparel": {
                "key": "size_apparel",
                "description": "Clothing/apparel size",
                "examples": "S, M, L, XL, 2XL, size 10, 10.5"
            },
            "Capacity": {
                "key": "capacity",
                "description": "Load or volume capacity",
                "examples": "500 lbs, 50 gallons, 1000 kg, 100L"
            },
            "Safety_Standard": {
                "key": "safety_standard",
                "description": "Safety compliance standard",
                "examples": "ANSI Z87.1, OSHA, EN 166, CSA Z94.3"
            },
            "Hazard_Class": {
                "key": "hazard_class",
                "description": "Hazardous area classification",
                "examples": "Class I Div 1, Zone 0, Zone 1, non-sparking"
            },
            "Protection_Level": {
                "key": "protection_level",
                "description": "Protection rating or level",
                "examples": "Level A4, cut resistant, flame retardant, FR"
            }
        }
    },
    
    "General": {
        "icon": "📝",
        "display_name": "General",
        "description": "General-purpose attributes",
        "attributes": {
            "Country_Origin": {
                "key": "country_origin",
                "description": "Country of manufacture",
                "examples": "USA, Germany, China, Japan, Made in USA"
            },
            "Shelf_Life": {
                "key": "shelf_life",
                "description": "Product shelf life or expiration",
                "examples": "2 years, 24 months, 5 year shelf life"
            },
            "Warranty": {
                "key": "warranty",
                "description": "Warranty period",
                "examples": "1 year, 2 year warranty, lifetime, 90 days"
            },
            "Certification": {
                "key": "certification",
                "description": "General certifications",
                "examples": "ISO 9001, NSF, FDA approved, RoHS, REACH"
            },
            "Other": {
                "key": "other",
                "description": "Any other important attribute",
                "examples": "special features, custom specifications"
            }
        }
    }
}


def get_all_attributes() -> List[str]:
    """Get flat list of all attribute names."""
    all_attrs = []
    for category in ATTRIBUTE_CATEGORIES.values():
        all_attrs.extend(category['attributes'].keys())
    return all_attrs


def get_category_attributes(category_key: str) -> List[str]:
    """Get attribute names for a specific category."""
    if category_key in ATTRIBUTE_CATEGORIES:
        return list(ATTRIBUTE_CATEGORIES[category_key]['attributes'].keys())
    return []


def get_attribute_info(attr_name: str) -> Dict[str, str]:
    """Get info for a specific attribute."""
    for category in ATTRIBUTE_CATEGORIES.values():
        if attr_name in category['attributes']:
            return category['attributes'][attr_name]
    return {}


def build_prompt_section(selected_attributes: List[str], custom_attributes: Dict[str, Dict] = None) -> str:
    """Build the attributes section of the AI prompt based on selected attributes."""
    
    if not selected_attributes and not custom_attributes:
        return ""
    
    lines = ["10. **Attributes** (extract if present in description, leave empty if not found):"]
    
    # Add selected predefined attributes
    for attr_name in selected_attributes:
        info = get_attribute_info(attr_name)
        if info:
            key = info['key']
            desc = info['description']
            examples = info['examples']
            lines.append(f"    - {key}: {desc} (e.g., \"{examples.split(', ')[0]}\", \"{examples.split(', ')[1] if ', ' in examples else ''}\")")
    
    # Add custom attributes
    if custom_attributes:
        for attr_name, info in custom_attributes.items():
            key = attr_name.lower().replace(' ', '_')
            desc = info.get('description', f'Extract {attr_name}')
            examples = info.get('examples', '')
            if examples:
                lines.append(f"    - {key}: {desc} (e.g., \"{examples}\")")
            else:
                lines.append(f"    - {key}: {desc}")
    
    lines.append("")
    lines.append("    Return as JSON object with lowercase attribute names using underscores as keys.")
    lines.append("    Only include attributes that are clearly mentioned or strongly implied in the description.")
    lines.append("    Leave empty string \"\" for attributes not found or not applicable.")
    
    return "\n".join(lines)


def get_attribute_key_map(selected_attributes: List[str], custom_attributes: Dict[str, Dict] = None) -> Dict[str, str]:
    """Build mapping from column names to JSON keys for selected attributes."""
    key_map = {}
    
    # Add predefined attributes
    for attr_name in selected_attributes:
        info = get_attribute_info(attr_name)
        if info:
            key_map[attr_name] = info['key']
    
    # Add custom attributes
    if custom_attributes:
        for attr_name in custom_attributes:
            key_map[attr_name] = attr_name.lower().replace(' ', '_')
    
    return key_map


# Default selections for quick presets
PRESET_SELECTIONS = {
    "Industrial_MRO": [
        "Material", "Size_Dimension", "Thread_Type", "Pressure_Rating", 
        "Temperature_Rating", "Part_Model_Number", "UOM"
    ],
    "Electrical": [
        "Voltage", "Power", "Current", "Phase", "IP_Rating"
    ],
    "IT_Technology": [
        "Part_Model_Number", "Processor", "RAM", "Storage", "Screen_Size", "Software"
    ],
    "Services": [
        "Service_Type", "Duration", "Scope", "SLA"
    ],
    "Safety_PPE": [
        "Material", "Size_Apparel", "Color", "Safety_Standard", "Protection_Level"
    ],
    "Full_MRO": [
        "Material", "Material_Grade", "Size_Dimension", "Thread_Type", 
        "Pressure_Rating", "Temperature_Rating", "Part_Model_Number",
        "Connection_Type", "Finish", "UOM", "Quantity_Per_Pack"
    ]
}


if __name__ == "__main__":
    # Test the configuration
    print("=" * 60)
    print("ATTRIBUTE CONFIGURATION TEST")
    print("=" * 60)
    
    print(f"\nTotal categories: {len(ATTRIBUTE_CATEGORIES)}")
    print(f"Total attributes: {len(get_all_attributes())}")
    
    print("\n--- Categories ---")
    for key, cat in ATTRIBUTE_CATEGORIES.items():
        print(f"{cat['icon']} {cat['display_name']}: {len(cat['attributes'])} attributes")
    
    print("\n--- Sample Prompt Section (Industrial preset) ---")
    sample_attrs = PRESET_SELECTIONS['Industrial_MRO']
    print(build_prompt_section(sample_attrs))
