"""
SRE (Software Reverse Engineering) Taxonomy Module
Defines the taxonomy structure and classification rules for SMS
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import spacy
from spacy.matcher import PhraseMatcher
import yaml
import logging
from pathlib import Path
import subprocess

@dataclass
class TaxonomyCategory:
    name: str
    keywords: Dict[str, List[str]]  # Language code -> keywords
    subcategories: Dict[str, Dict[str, List[str]]]  # Language code -> subcategories

class SRETaxonomy:
    def __init__(self, default_lang='en'):
        self.logger = logging.getLogger(__name__)
        self.default_lang = default_lang
        self.supported_langs = {'en', 'fr'}
        self.nlp_models = {}
        self.matchers = {'en': {}, 'fr': {}}
        
        self._initialize_taxonomy()
        self._initialize_nlp()
        
    def _initialize_taxonomy(self):
        """Initialize multilingual taxonomy structure"""
        self.categories = {
            "methods": TaxonomyCategory(
                name="Methods",
                keywords={
                    'en': ["reverse engineering", "software reverse engineering", "SRE"],
                    'fr': ["rétro-ingénierie", "ingénierie inverse", "rétro-conception"]
                },
                subcategories={
                    'en': {
                        "static": ["static analysis", "code analysis", "decompilation"],
                        "dynamic": ["dynamic analysis", "runtime analysis", "execution trace"],
                        "hybrid": ["hybrid analysis", "multi-modal analysis"]
                    },
                    'fr': {
                        "static": ["analyse statique", "analyse de code", "décompilation"],
                        "dynamic": ["analyse dynamique", "analyse d'exécution", "trace d'exécution"],
                        "hybrid": ["analyse hybride", "analyse multi-modale"]
                    }
                }
            ),
            "applications": TaxonomyCategory(
                name="Applications",
                keywords={
                    'en': ["application", "domain", "use case"],
                    'fr': ["application", "domaine", "cas d'utilisation"]
                },
                subcategories={
                    'en': {
                        "security": ["malware analysis", "vulnerability detection", "security assessment"],
                        "maintenance": ["software maintenance", "legacy system", "program comprehension"],
                        "ai_integration": ["AI-assisted", "machine learning", "LLM", "neural"]
                    },
                    'fr': {
                        "security": ["analyse de malware", "détection de vulnérabilité", "évaluation de sécurité"],
                        "maintenance": ["maintenance logicielle", "système legacy", "compréhension de programme"],
                        "ai_integration": ["assisté par IA", "apprentissage automatique", "LLM", "neural"]
                    }
                }
            ),
            "tools": TaxonomyCategory(
                name="Tools",
                keywords={
                    'en': ["tool", "framework", "platform"],
                    'fr': ["outil", "framework", "plateforme"]
                },
                subcategories={
                    'en': {
                        "decompilers": ["decompiler", "disassembler", "binary analysis tool"],
                        "debuggers": ["debugger", "dynamic analysis tool"],
                        "analyzers": ["static analyzer", "code analysis tool"]
                    },
                    'fr': {
                        "decompilers": ["décompilateur", "désassembleur", "outil d'analyse binaire"],
                        "debuggers": ["débogueur", "outil d'analyse dynamique"],
                        "analyzers": ["analyseur statique", "outil d'analyse de code"]
                    }
                }
            )
        }
        
    def _initialize_nlp(self):
        """Initialize NLP models for all supported languages"""
        model_mapping = {
            'en': 'en_core_web_md',
            'fr': 'fr_core_news_md'
        }
        
        for lang, model_name in model_mapping.items():
            try:
                self.nlp_models[lang] = spacy.load(model_name)
                self.logger.info(f"Loaded {model_name} for {lang}")
            except OSError:
                self.logger.warning(f"Downloading {model_name}...")
                subprocess.run(["python", "-m", "spacy", "download", model_name])
                self.nlp_models[lang] = spacy.load(model_name)
            
            # Initialize matchers for each language
            self._initialize_matchers(lang)
    
    def _initialize_matchers(self, lang):
        """Initialize pattern matchers for a specific language"""
        nlp = self.nlp_models[lang]
        
        for category, data in self.categories.items():
            matcher = PhraseMatcher(nlp.vocab)
            patterns = []
            
            # Add main category keywords
            patterns.extend([nlp(text.lower()) for text in data.keywords[lang]])
            
            # Add subcategory keywords
            for subcat_keywords in data.subcategories[lang].values():
                patterns.extend([nlp(text.lower()) for text in subcat_keywords])
                
            matcher.add(category, None, *patterns)
            self.matchers[lang][category] = matcher
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        # Simple language detection based on character frequency
        # You might want to use a more sophisticated method like langdetect
        text = text.lower()
        fr_chars = set('éèêëàâäôöûüçïîœæ')
        fr_score = sum(1 for c in text if c in fr_chars)
        
        return 'fr' if fr_score > 0 else 'en'
    
    def classify_paper(self, title: str, abstract: str = "") -> Dict[str, Set[str]]:
        """Classify a paper according to the SRE taxonomy"""
        classifications = {cat: set() for cat in self.categories}
        
        # Combine title and abstract for analysis
        text = f"{title} {abstract}".lower()
        
        # Detect language or use default
        lang = self.detect_language(text) if text.strip() else self.default_lang
        if lang not in self.supported_langs:
            lang = self.default_lang
            
        doc = self.nlp_models[lang](text)
        
        # Match against each category
        for category, matcher in self.matchers[lang].items():
            matches = matcher(doc)
            if matches:
                cat_data = self.categories[category]
                # Determine subcategories
                for subcat, keywords in cat_data.subcategories[lang].items():
                    if any(keyword.lower() in text for keyword in keywords):
                        classifications[category].add(subcat)
                        
        return classifications

    def get_taxonomy_stats(self, papers_classifications: List[Dict[str, Set[str]]]) -> Dict:
        """Generate statistics about taxonomy distribution"""
        stats = {
            "total_papers": len(papers_classifications),
            "categories": {}
        }
        
        for category in self.categories:
            cat_stats = {
                "total": 0,
                "subcategories": {
                    subcat: 0 
                    for subcat in self.categories[category].subcategories[self.default_lang]
                }
            }
            
            for paper_class in papers_classifications:
                if paper_class[category]:
                    cat_stats["total"] += 1
                    for subcat in paper_class[category]:
                        cat_stats["subcategories"][subcat] += 1
                        
            stats["categories"][category] = cat_stats
            
        return stats
