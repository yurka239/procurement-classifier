"""Normalization Module for Procurement Classifier v2.1

Handles post-processing normalization using fingerprinting and clustering.
Converts variations like "Ball Valve", "ball valve", "BALL VLV" into canonical forms.
"""

import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
from difflib import SequenceMatcher


class Fingerprinter:
    """Text fingerprinting for clustering similar values."""
    
    def __init__(self, ngram_size: int = 2):
        self.ngram_size = ngram_size
    
    def key_fingerprint(self, text: str) -> str:
        """
        OpenRefine-compatible key collision fingerprint.
        
        Steps:
        1. Strip and lowercase
        2. Normalize unicode (remove accents)
        3. Remove punctuation (replace with space)
        4. Collapse multiple spaces
        5. Split, sort unique tokens, rejoin
        
        Example: "Ball Valve, SS" -> "ball ss valve"
        Example: "call-out" -> "call out"
        """
        if not text:
            return ""
        
        # Strip and lowercase
        text = str(text).strip().lower()
        
        # Normalize unicode (remove accents)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Remove punctuation, replace with space
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Split, sort unique tokens, rejoin
        tokens = sorted(set(text.split()))
        
        return ' '.join(tokens)
    
    def ngram_fingerprint(self, text: str) -> Set[str]:
        """Create n-gram fingerprint (set of character n-grams)."""
        if not text:
            return set()
        
        # Preprocess
        text = text.lower()
        text = re.sub(r'[^a-z0-9]', '', text)
        
        if len(text) < self.ngram_size:
            return {text}
        
        # Generate n-grams
        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        
        return ngrams
    
    def ngram_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between n-gram fingerprints (0.0-1.0)."""
        set1 = self.ngram_fingerprint(text1)
        set2 = self.ngram_fingerprint(text2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def sequence_similarity(self, text1: str, text2: str) -> float:
        """Calculate sequence similarity using SequenceMatcher."""
        if not text1 or not text2:
            return 0.0
        
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


class ValueClusterer:
    """Clusters similar values and selects canonical forms."""
    
    def __init__(
        self, 
        fingerprinter: Fingerprinter,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 1
    ):
        self.fingerprinter = fingerprinter
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
    
    def cluster_values(self, values: List[str]) -> Dict[str, str]:
        """
        Cluster similar values and return mapping to canonical forms.
        
        Args:
            values: List of raw values to cluster
        
        Returns:
            Dict mapping each original value to its canonical form
        """
        if not values:
            return {}
        
        # Count occurrences
        value_counts = Counter(v.strip() for v in values if v and str(v).strip() and str(v).lower() != 'nan')
        unique_values = list(value_counts.keys())
        
        if len(unique_values) <= 1:
            canonical = unique_values[0] if unique_values else ""
            return {v: canonical for v in values if v and str(v).strip()}
        
        # Phase 1: Group by key fingerprint (exact structural match)
        key_groups = defaultdict(list)
        for value in unique_values:
            key = self.fingerprinter.key_fingerprint(value)
            key_groups[key].append(value)
        
        # Phase 2: Merge similar key groups using n-gram similarity
        merged_clusters = []
        processed_keys = set()
        
        keys = list(key_groups.keys())
        for i, key1 in enumerate(keys):
            if key1 in processed_keys:
                continue
            
            cluster = list(key_groups[key1])
            processed_keys.add(key1)
            
            # Try to merge with other groups
            for j, key2 in enumerate(keys[i + 1:], i + 1):
                if key2 in processed_keys:
                    continue
                
                # Check similarity
                sim = self.fingerprinter.ngram_similarity(key1, key2)
                if sim >= self.similarity_threshold:
                    cluster.extend(key_groups[key2])
                    processed_keys.add(key2)
            
            merged_clusters.append(cluster)
        
        # Phase 3: Select canonical form for each cluster
        mapping = {}
        for cluster in merged_clusters:
            canonical = self._select_canonical(cluster, value_counts)
            for value in cluster:
                mapping[value] = canonical
        
        # Apply mapping to all original values
        result = {}
        for value in values:
            if value and str(value).strip() and str(value).lower() != 'nan':
                stripped = str(value).strip()
                result[value] = mapping.get(stripped, stripped)
        
        return result
    
    def _select_canonical(self, cluster: List[str], counts: Counter) -> str:
        """
        Select the best canonical form from a cluster.
        
        Priority:
        1. Most frequent
        2. If tie: lowercase version
        3. If tie: shortest
        """
        if not cluster:
            return ""
        
        if len(cluster) == 1:
            return cluster[0].lower()
        
        # Score each candidate
        scored = []
        for value in cluster:
            freq = counts.get(value, 0)
            is_lower = value == value.lower()
            length = len(value)
            # Higher frequency is better, lowercase preferred, shorter preferred
            score = (freq * 1000, is_lower, -length)
            scored.append((score, value))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Return canonical in lowercase
        return scored[0][1].lower()


class AttributeNormalizer:
    """Normalize extracted attributes for benchmarking."""
    
    def __init__(self):
        # Material abbreviation mappings
        self.material_map = {
            'ss': 'stainless steel',
            'st steel': 'stainless steel',
            'st. steel': 'stainless steel',
            'stainless': 'stainless steel',
            'cs': 'carbon steel',
            'ms': 'mild steel',
            'al': 'aluminum',
            'alu': 'aluminum',
            'cu': 'copper',
            'br': 'brass',
            'bronze': 'bronze',
            'pvc': 'pvc',
            'pp': 'polypropylene',
            'pe': 'polyethylene',
            'ptfe': 'ptfe',
            'teflon': 'ptfe',
            'rubber': 'rubber',
            'steel': 'steel'
        }
        
        # Thread type mappings
        self.thread_map = {
            'npt': 'NPT',
            'bsp': 'BSP',
            'bspt': 'BSPT',
            'bspp': 'BSPP',
            'metric': 'Metric',
            'unf': 'UNF',
            'unc': 'UNC'
        }
    
    def normalize_material(self, material: str) -> str:
        """Normalize material name."""
        if not material or str(material).lower() == 'nan':
            return ''
        
        material_lower = str(material).lower().strip()
        return self.material_map.get(material_lower, material_lower)
    
    def normalize_color(self, color: str) -> str:
        """Normalize color/finish."""
        if not color or str(color).lower() == 'nan':
            return ''
        
        return str(color).lower().strip()
    
    def normalize_size(self, size_str: str) -> str:
        """Normalize size to standard format."""
        if not size_str or str(size_str).lower() == 'nan':
            return ''
        
        size_str = str(size_str).lower().strip()
        
        # Convert fractions to decimal: "1/2in" -> "0.5in"
        fraction_match = re.match(r'(\d+)/(\d+)\s*(\w+)?', size_str)
        if fraction_match:
            num, denom, unit = fraction_match.groups()
            decimal = float(num) / float(denom)
            unit = unit or 'in'
            return f"{decimal}{unit}"
        
        # Standardize units
        size_str = re.sub(r'\s*(inch|inches)\b', 'in', size_str)
        size_str = re.sub(r'\s*(millimeter|millimeters)\b', 'mm', size_str)
        size_str = re.sub(r'\s*(centimeter|centimeters)\b', 'cm', size_str)
        
        return size_str
    
    def normalize_thread(self, thread: str) -> str:
        """Normalize thread type."""
        if not thread or str(thread).lower() == 'nan':
            return ''
        
        thread_lower = str(thread).lower().strip()
        return self.thread_map.get(thread_lower, str(thread).upper())


class ProcurementNormalizer:
    """Main normalizer for procurement data."""
    
    def __init__(
        self,
        fingerprint_threshold: float = 0.85,
        ngram_size: int = 2,
        min_cluster_size: int = 2
    ):
        self.fingerprinter = Fingerprinter(ngram_size=ngram_size)
        self.clusterer = ValueClusterer(
            self.fingerprinter,
            similarity_threshold=fingerprint_threshold,
            min_cluster_size=min_cluster_size
        )
        self.attr_normalizer = AttributeNormalizer()
        
        # Storage for learned patterns
        self.noun_mappings: Dict[str, str] = {}
        self.category_mappings: Dict[str, str] = {}
        self.brand_mappings: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            'nouns_clustered': 0,
            'categories_clustered': 0,
            'brands_clustered': 0
        }
    
    def normalize_column(
        self, 
        values: List[str], 
        column_type: str = 'generic'
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Normalize a column of values.
        
        Args:
            values: List of raw values
            column_type: 'noun', 'category', 'brand', or 'generic'
        
        Returns:
            Tuple of (mapping dict, list of canonical values in order)
        """
        # Get clustering mapping
        mapping = self.clusterer.cluster_values(values)
        
        # Apply type-specific processing
        if column_type == 'noun':
            mapping = self._process_nouns(mapping)
            self.noun_mappings.update(mapping)
        elif column_type == 'category':
            mapping = self._process_categories(mapping)
            self.category_mappings.update(mapping)
        elif column_type == 'brand':
            mapping = self._process_brands(mapping)
            self.brand_mappings.update(mapping)
        
        # Count unique clusters
        unique_canonicals = len(set(mapping.values()))
        original_unique = len(set(str(v).strip().lower() for v in values if v and str(v).strip() and str(v).lower() != 'nan'))
        clustered_count = max(0, original_unique - unique_canonicals)
        
        if column_type == 'noun':
            self.stats['nouns_clustered'] = clustered_count
        elif column_type == 'category':
            self.stats['categories_clustered'] = clustered_count
        elif column_type == 'brand':
            self.stats['brands_clustered'] = clustered_count
        
        # Return normalized values in original order
        normalized = [mapping.get(v, v) if v and str(v).strip() and str(v).lower() != 'nan' else '' for v in values]
        
        return mapping, normalized
    
    def _process_nouns(self, mapping: Dict[str, str]) -> Dict[str, str]:
        """Apply noun-specific normalization rules."""
        processed = {}
        for original, canonical in mapping.items():
            # Ensure lowercase
            canonical = canonical.lower()
            
            # Singularize common patterns
            canonical = self._simple_singularize(canonical)
            
            processed[original] = canonical
        
        return processed
    
    def _process_categories(self, mapping: Dict[str, str]) -> Dict[str, str]:
        """Apply category-specific normalization rules."""
        processed = {}
        for original, canonical in mapping.items():
            # Lowercase
            canonical = canonical.lower()
            
            # Remove redundant words
            canonical = re.sub(r'\b(category|type|class)\b', '', canonical).strip()
            canonical = re.sub(r'\s+', ' ', canonical)
            
            processed[original] = canonical
        
        return processed
    
    def _process_brands(self, mapping: Dict[str, str]) -> Dict[str, str]:
        """Apply brand-specific normalization rules."""
        processed = {}
        for original, canonical in mapping.items():
            # Lowercase for consistency
            canonical = canonical.lower()
            
            # Remove common suffixes
            canonical = re.sub(r'\s*(inc\.?|ltd\.?|llc|corp\.?|co\.?|gmbh|ag|sa)$', '', canonical, flags=re.I)
            canonical = canonical.strip()
            
            processed[original] = canonical
        
        return processed
    
    def _simple_singularize(self, word: str) -> str:
        """Simple rule-based singularization."""
        if len(word) <= 3:
            return word
        
        # Don't singularize these
        exceptions = {'series', 'species', 'glasses', 'pliers', 'scissors', 'headquarters'}
        if word in exceptions:
            return word
        
        # Common plural patterns
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'
        elif word.endswith('ves') and len(word) > 4:
            return word[:-3] + 'f'
        elif word.endswith('es') and len(word) > 3:
            root = word[:-2]
            if root.endswith(('s', 'x', 'z', 'ch', 'sh')):
                return root
            return word[:-1]
        elif word.endswith('s') and not word.endswith('ss'):
            return word[:-1]
        
        return word
    
    def generate_normalized_description(
        self,
        concept_noun: str,
        original_text: str,
        brand: str = '',
        modifier: str = '',
        attributes: Dict[str, str] = None
    ) -> str:
        """
        Generate a clean, normalized description for benchmarking.
        
        Format: concept_noun - modifier(s) ; brand ; key_attributes
        Examples:
        - "bearing - ball ; sealed ; skf ; 25x52x15 mm ; C3 clearance"
        - "valve - ball ; stainless steel ; parker ; 3/4 in ; PN16 ; threaded BSPT"
        - "pump - submersible ; slurry ; grundfos ; 5 kW ; 230 V ; 50 Hz"
        """
        if not concept_noun:
            return original_text.lower() if original_text else ''
        
        parts = []
        
        # Start with concept noun
        parts.append(concept_noun.lower())
        
        # Add modifier(s)
        if modifier:
            # Clean modifier: remove extra spaces, lowercase, split by comma
            modifiers = [m.strip().lower() for m in modifier.split(',') if m.strip()]
            parts.extend(modifiers)
        
        # Add brand
        if brand:
            parts.append(brand.lower())
        
        # Add key attributes
        if attributes:
            # Priority order for attributes - include ALL available attributes
            priority_attrs = [
                'size_primary', 'size_secondary', 'size_unit',
                'material', 'material_grade', 'color',
                'thread_type', 'pressure_rating', 'temperature_rating',
                'voltage', 'power', 'weight', 'certification',
                'quantity_per_pack'
            ]
            for attr_key in priority_attrs:
                if attr_key in attributes and attributes[attr_key]:
                    val = str(attributes[attr_key]).lower()
                    # Skip if already included (e.g., size_unit might be in size_primary)
                    if val not in parts:
                        parts.append(val)
        
        # If we have very few parts (< 3) and original text is short, include key terms from original
        # This helps preserve information for services and simple items
        if len(parts) < 3 and original_text:
            # Extract key terms from original (skip common words)
            original_lower = original_text.lower()
            skip_words = {'the', 'a', 'an', 'and', 'or', 'for', 'to', 'of', 'in', 'on', 'at', 'by', 'with'}
            words = [w.strip('.,;:()[]{}"\'-') for w in original_lower.split()]
            key_terms = [w for w in words if len(w) > 3 and w not in skip_words and w not in parts]
            # Add up to 3 key terms
            for term in key_terms[:3]:
                if term and term not in parts:
                    parts.append(term)
        
        # Join parts: first separator is " - ", rest are " ; "
        if len(parts) <= 1:
            return parts[0] if parts else ''
        else:
            return parts[0] + ' - ' + ' ; '.join(parts[1:])
    
    def save_patterns(self, filepath: Path) -> None:
        """Save learned normalization patterns to JSON."""
        patterns = {
            'noun_mappings': self.noun_mappings,
            'category_mappings': self.category_mappings,
            'brand_mappings': self.brand_mappings,
            'stats': self.stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)
        
        print(f"[Normalizer] Saved patterns to {filepath}")
    
    def load_patterns(self, filepath: Path) -> None:
        """Load previously learned patterns."""
        if not filepath.exists():
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
            
            self.noun_mappings = patterns.get('noun_mappings', {})
            self.category_mappings = patterns.get('category_mappings', {})
            self.brand_mappings = patterns.get('brand_mappings', {})
            
            print(f"[Normalizer] Loaded {len(self.noun_mappings)} noun mappings")
            print(f"[Normalizer] Loaded {len(self.category_mappings)} category mappings")
            print(f"[Normalizer] Loaded {len(self.brand_mappings)} brand mappings")
            
        except Exception as e:
            print(f"[Normalizer] Error loading patterns: {e}")
    
    def get_summary(self) -> Dict[str, any]:
        """Get normalization summary statistics."""
        return {
            'unique_nouns': len(set(self.noun_mappings.values())),
            'unique_categories': len(set(self.category_mappings.values())),
            'unique_brands': len(set(self.brand_mappings.values())),
            'nouns_clustered': self.stats.get('nouns_clustered', 0),
            'categories_clustered': self.stats.get('categories_clustered', 0),
            'brands_clustered': self.stats.get('brands_clustered', 0)
        }
