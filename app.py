#!/usr/bin/env python3
"""
LinkedIn Profile Search Engine with OpenSearch
Natural Language to Search Implementation

Features:
1. Load JSONL data into OpenSearch
2. Natural language query processing
3. Semantic search with vector embeddings
4. Traditional keyword search
5. Hybrid search combining both approaches
6. Interactive CLI interface

Requirements:
pip install opensearch-py sentence-transformers nltk spacy
python -m spacy download en_core_web_sm
"""

import json
import re
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
import nltk
import spacy
from dataclasses import dataclass
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Structure for search results"""
    score: float
    profile: Dict[str, Any]
    highlight: Optional[Dict[str, List[str]]] = None

class LinkedInSearchEngine:
    def __init__(self, host='localhost', port=9200):
        """Initialize the search engine"""
        self.client = OpenSearch([{'host': host, 'port': port}])
        self.index_name = 'linkedin_profiles'
        
        # Load NLP models
        print("Loading NLP models...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_sm')
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
            
        print(" Search engine initialized!")

    def create_index(self):
        """Create OpenSearch index with proper mappings"""
        index_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "custom_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "first_name": {"type": "text", "analyzer": "custom_analyzer"},
                    "last_name": {"type": "text", "analyzer": "custom_analyzer"},
                    "title": {"type": "text", "analyzer": "custom_analyzer"},
                    "location": {"type": "text", "analyzer": "custom_analyzer"},
                    "city": {"type": "keyword"},
                    "country": {"type": "keyword"},
                    "current_industry": {"type": "text", "analyzer": "custom_analyzer"},
                    "functional_area": {"type": "keyword"},
                    "seniority_level": {"type": "keyword"},
                    "expertise": {"type": "text", "analyzer": "custom_analyzer"},
                    "summary": {"type": "text", "analyzer": "custom_analyzer"},
                    "experience": {
                        "type": "nested",
                        "properties": {
                            "name": {"type": "text", "analyzer": "custom_analyzer"},
                            "industry": {"type": "text", "analyzer": "custom_analyzer"},
                            "country": {"type": "keyword"}
                        }
                    },
                    "education": {
                        "type": "nested",
                        "properties": {
                            "campus": {"type": "text", "analyzer": "custom_analyzer"},
                            "major": {"type": "text", "analyzer": "custom_analyzer"},
                            "specialization": {"type": "text", "analyzer": "custom_analyzer"}
                        }
                    },
                    "certifications": {
                        "type": "nested",
                        "properties": {
                            "title": {"type": "text", "analyzer": "custom_analyzer"},
                            "description": {"type": "text", "analyzer": "custom_analyzer"}
                        }
                    },
                    # Vector field for semantic search
                    "profile_embedding": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    # Combined text for search
                    "searchable_text": {"type": "text", "analyzer": "custom_analyzer"}
                }
            }
        }
        
        # Delete index if exists
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            print(f" Deleted existing index: {self.index_name}")
        
        # Create new index
        self.client.indices.create(index=self.index_name, body=index_body)
        print(f" Created index: {self.index_name}")

    def process_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single profile for indexing"""
        # Handle "NA" values
        def clean_na(value):
            if value == "NA" or value == ["NA"]:
                return None if isinstance(value, str) else []
            return value
        
        # Clean the profile
        processed = {}
        for key, value in profile.items():
            processed[key] = clean_na(value)
        
        # Create searchable text combining key fields
        searchable_parts = []
        
        # Add basic info
        if processed.get('first_name'):
            searchable_parts.append(processed['first_name'])
        if processed.get('last_name'):
            searchable_parts.append(processed['last_name'])
        if processed.get('title'):
            searchable_parts.append(processed['title'])
        if processed.get('summary'):
            searchable_parts.append(processed['summary'])
        if processed.get('expertise'):
            searchable_parts.append(processed['expertise'])
        if processed.get('current_industry'):
            searchable_parts.append(processed['current_industry'])
        
        # Add experience company names and industries
        if processed.get('experience') and isinstance(processed['experience'], list):
            for exp in processed['experience']:
                if isinstance(exp, dict):
                    if exp.get('name'):
                        searchable_parts.append(exp['name'])
                    if exp.get('industry'):
                        searchable_parts.append(exp['industry'])
        
        # Add education
        if processed.get('education') and isinstance(processed['education'], list):
            for edu in processed['education']:
                if isinstance(edu, dict):
                    if edu.get('campus'):
                        searchable_parts.append(edu['campus'])
                    if edu.get('major'):
                        searchable_parts.append(edu['major'])
                    if edu.get('specialization'):
                        searchable_parts.append(edu['specialization'])
        
        # Combine all searchable text
        searchable_text = ' '.join(filter(None, searchable_parts))
        processed['searchable_text'] = searchable_text
        
        # Generate embedding for semantic search
        if searchable_text:
            embedding = self.sentence_model.encode(searchable_text)
            processed['profile_embedding'] = embedding.tolist()
        
        return processed

    def load_jsonl_data(self, file_path: str, max_records: Optional[int] = None):
        """Load JSONL data into OpenSearch"""
        print(f" Loading data from {file_path}...")
        
        def generate_docs():
            with open(file_path, 'r', encoding='utf-8') as file:
                count = 0
                for line_num, line in enumerate(file, 1):
                    try:
                        if max_records and count >= max_records:
                            break
                            
                        profile = json.loads(line.strip())
                        processed_profile = self.process_profile(profile)
                        
                        doc = {
                            "_index": self.index_name,
                            "_id": f"profile_{line_num}",
                            "_source": processed_profile
                        }
                        yield doc
                        count += 1
                        
                        if count % 1000 == 0:
                            print(f" Processed {count} profiles...")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line {line_num}: {e}")
                        continue
        
        # Bulk index the documents
        try:
            helpers.bulk(self.client, generate_docs(), chunk_size=500)
            
            # Refresh index
            self.client.indices.refresh(index=self.index_name)
            
            # Get count
            count_response = self.client.count(index=self.index_name)
            total_docs = count_response['count']
            
            print(f" Successfully indexed {total_docs} profiles!")
            
        except Exception as e:
            logger.error(f"Error during bulk indexing: {e}")
            raise

    def parse_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query into structured search parameters"""
        doc = self.nlp(query.lower())
        
        parsed = {
            'skills': [],
            'location': [],
            'job_titles': [],
            'companies': [],
            'experience_level': None,
            'education': [],
            'industries': [],
            'raw_query': query
        }
        
        # Skill-related keywords
        skill_patterns = [
            r'\b(python|java|javascript|react|angular|vue|node\.?js|django|flask|sql|mysql|postgresql|mongodb|redis|aws|azure|gcp|docker|kubernetes|git|machine learning|ml|ai|artificial intelligence|data science|tensorflow|pytorch|pandas|numpy|scikit-learn|opencv|nlp|natural language processing|computer vision|deep learning|neural networks|blockchain|devops|ci/cd|jenkins|terraform|ansible|linux|unix|windows|html|css|sass|less|bootstrap|tailwind|figma|photoshop|illustrator|sketch|ui/ux|frontend|backend|full.?stack|api|rest|graphql|microservices|agile|scrum|kanban|jira|confluence|slack|teams|zoom|excel|powerbi|tableau|power bi|salesforce|hubspot|magento|wordpress|shopify|seo|sem|google analytics|facebook ads|linkedin ads|social media|content marketing|email marketing|affiliate marketing|ppc|cro|a/b testing)\b'
        ]
        
        # Location patterns
        location_patterns = [
            r'\b(bangalore|bengaluru|mumbai|delhi|new delhi|hyderabad|chennai|pune|kolkata|ahmedabad|surat|lucknow|kanpur|jaipur|indore|bhopal|patna|vadodara|ludhiana|agra|nashik|faridabad|meerut|rajkot|kalyan|vasai virar|varanasi|srinagar|aurangabad|dhanbad|amritsar|navi mumbai|allahabad|ranchi|haora|coimbatore|jabalpur|gwalior|vijayawada|jodhpur|madurai|raipur|kota|guwahati|chandigarh|solapur|hubballi dharwad|tiruchirappalli|bareilly|moradabad|mysore|tiruppur|gurgaon|gurugram|noida|faridabad|ghaziabad|greater noida)\b'
        ]
        
        # Experience level patterns
        experience_patterns = {
            'entry': r'\b(entry.?level|junior|fresher|graduate|trainee|intern|0.?2 years?|fresh|beginner)\b',
            'mid': r'\b(mid.?level|senior|experienced|3.?7 years?|4.?6 years?)\b',
            'senior': r'\b(senior|lead|principal|architect|manager|director|7\+ years?|8\+ years?|10\+ years?)\b',
            'executive': r'\b(executive|vp|vice president|ceo|cto|cfo|head|director)\b'
        }
        
        # Job title patterns
        job_title_patterns = [
            r'\b(software engineer|developer|programmer|architect|tech lead|engineering manager|product manager|data scientist|data analyst|data engineer|ml engineer|devops engineer|frontend developer|backend developer|full.?stack developer|ui/ux designer|product designer|marketing manager|sales manager|business analyst|project manager|scrum master|qa engineer|test engineer|site reliability engineer|cloud engineer|security engineer|mobile developer|ios developer|android developer|web developer|system administrator|database administrator|network engineer|cyber security|information security|digital marketing|content writer|copywriter|seo specialist|social media manager|hr manager|recruiter|business development|account manager|customer success)\b'
        ]
        
        query_lower = query.lower()
        
        # Extract skills
        for pattern in skill_patterns:
            matches = re.findall(pattern, query_lower)
            parsed['skills'].extend(matches)
        
        # Extract locations
        for pattern in location_patterns:
            matches = re.findall(pattern, query_lower)
            parsed['location'].extend(matches)
        
        # Extract experience level
        for level, pattern in experience_patterns.items():
            if re.search(pattern, query_lower):
                parsed['experience_level'] = level
                break
        
        # Extract job titles
        for pattern in job_title_patterns:
            matches = re.findall(pattern, query_lower)
            parsed['job_titles'].extend(matches)
        
        # Use spaCy for entities
        for ent in doc.ents:
            if ent.label_ in ['ORG']:  # Organizations (companies)
                parsed['companies'].append(ent.text)
            elif ent.label_ in ['GPE']:  # Geopolitical entities (locations)
                parsed['location'].append(ent.text)
        
        # Clean up duplicates
        for key in ['skills', 'location', 'job_titles', 'companies', 'education', 'industries']:
            parsed[key] = list(set(parsed[key]))
        
        return parsed

    def build_search_query(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Build OpenSearch query from parsed natural language"""
        must_clauses = []
        should_clauses = []
        
        # Skills search
        if parsed_query['skills']:
            skills_query = {
                "bool": {
                    "should": [
                        {"match": {"expertise": " ".join(parsed_query['skills'])}},
                        {"match": {"searchable_text": " ".join(parsed_query['skills'])}}
                    ]
                }
            }
            must_clauses.append(skills_query)
        
        # Location search
        if parsed_query['location']:
            location_query = {
                "bool": {
                    "should": [
                        {"match": {"location": " ".join(parsed_query['location'])}},
                        {"match": {"city": " ".join(parsed_query['location'])}},
                        {"match": {"country": " ".join(parsed_query['location'])}}
                    ]
                }
            }
            must_clauses.append(location_query)
        
        # Job titles search
        if parsed_query['job_titles']:
            title_query = {
                "match": {
                    "title": " ".join(parsed_query['job_titles'])
                }
            }
            must_clauses.append(title_query)
        
        # Experience level
        if parsed_query['experience_level']:
            exp_query = {
                "match": {
                    "seniority_level": parsed_query['experience_level']
                }
            }
            should_clauses.append(exp_query)
        
        # Companies
        if parsed_query['companies']:
            company_query = {
                "nested": {
                    "path": "experience",
                    "query": {
                        "match": {
                            "experience.name": " ".join(parsed_query['companies'])
                        }
                    }
                }
            }
            should_clauses.append(company_query)
        
        # Fallback: full-text search on everything
        if parsed_query['raw_query']:
            fallback_query = {
                "multi_match": {
                    "query": parsed_query['raw_query'],
                    "fields": ["searchable_text^2", "title^1.5", "expertise", "summary"],
                    "fuzziness": "AUTO"
                }
            }
            should_clauses.append(fallback_query)
        
        # Construct final query
        if must_clauses or should_clauses:
            query = {
                "bool": {
                    "must": must_clauses if must_clauses else [],
                    "should": should_clauses,
                    "minimum_should_match": 1 if should_clauses and not must_clauses else 0
                }
            }
        else:
            query = {"match_all": {}}
        
        return query

    def semantic_search(self, query: str, size: int = 10) -> List[SearchResult]:
        """Perform semantic search using embeddings"""
        query_embedding = self.sentence_model.encode(query)
        
        search_body = {
            "size": size,
            "query": {
                "knn": {
                    "profile_embedding": {
                        "vector": query_embedding.tolist(),
                        "k": size
                    }
                }
            },
            "_source": {"excludes": ["profile_embedding"]}
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        
        results = []
        for hit in response['hits']['hits']:
            result = SearchResult(
                score=hit['_score'],
                profile=hit['_source']
            )
            results.append(result)
        
        return results

    def keyword_search(self, query: str, size: int = 10) -> List[SearchResult]:
        """Perform traditional keyword search"""
        parsed_query = self.parse_natural_language_query(query)
        search_query = self.build_search_query(parsed_query)
        
        search_body = {
            "size": size,
            "query": search_query,
            "highlight": {
                "fields": {
                    "searchable_text": {},
                    "title": {},
                    "expertise": {},
                    "summary": {}
                }
            },
            "_source": {"excludes": ["profile_embedding"]}
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        
        results = []
        for hit in response['hits']['hits']:
            result = SearchResult(
                score=hit['_score'],
                profile=hit['_source'],
                highlight=hit.get('highlight')
            )
            results.append(result)
        
        return results

    def hybrid_search(self, query: str, size: int = 10, semantic_weight: float = 0.3) -> List[SearchResult]:
        """Combine semantic and keyword search results"""
        # Get results from both approaches
        semantic_results = self.semantic_search(query, size * 2)
        keyword_results = self.keyword_search(query, size * 2)
        
        # Combine and re-rank
        combined_scores = {}
        
        # Add semantic scores
        for result in semantic_results:
            profile_id = result.profile.get('linkedin_url', str(id(result.profile)))
            combined_scores[profile_id] = {
                'semantic_score': result.score * semantic_weight,
                'keyword_score': 0,
                'profile': result.profile,
                'highlight': None
            }
        
        # Add keyword scores
        for result in keyword_results:
            profile_id = result.profile.get('linkedin_url', str(id(result.profile)))
            if profile_id in combined_scores:
                combined_scores[profile_id]['keyword_score'] = result.score * (1 - semantic_weight)
                combined_scores[profile_id]['highlight'] = result.highlight
            else:
                combined_scores[profile_id] = {
                    'semantic_score': 0,
                    'keyword_score': result.score * (1 - semantic_weight),
                    'profile': result.profile,
                    'highlight': result.highlight
                }
        
        # Calculate final scores and sort
        final_results = []
        for profile_id, scores in combined_scores.items():
            final_score = scores['semantic_score'] + scores['keyword_score']
            result = SearchResult(
                score=final_score,
                profile=scores['profile'],
                highlight=scores['highlight']
            )
            final_results.append(result)
        
        # Sort by final score and return top results
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:size]

    def search(self, query: str, method: str = 'hybrid', size: int = 10) -> List[SearchResult]:
        """Main search interface"""
        if method == 'semantic':
            return self.semantic_search(query, size)
        elif method == 'keyword':
            return self.keyword_search(query, size)
        elif method == 'hybrid':
            return self.hybrid_search(query, size)
        else:
            raise ValueError("Method must be 'semantic', 'keyword', or 'hybrid'")

    def format_result(self, result: SearchResult, index: int) -> str:
        """Format search result for display"""
        profile = result.profile
        
        # Basic info
        name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
        title = profile.get('title', 'N/A')
        location = profile.get('location', 'N/A')
        industry = profile.get('current_industry', 'N/A')
        
        # Skills
        expertise = profile.get('expertise', '')
        skills = expertise.split(',')[:5] if expertise else []
        skills_str = ', '.join(skills) if skills else 'N/A'
        
        # Experience
        experience = profile.get('experience', [])
        current_company = 'N/A'
        if experience and isinstance(experience, list) and len(experience) > 0:
            if isinstance(experience[0], dict):
                current_company = experience[0].get('name', 'N/A')
        
        output = f"""
{'='*60}
 Result #{index + 1} (Score: {result.score:.3f})
{'='*60}
 Name: {name}
 Title: {title}
 Company: {current_company}
 Location: {location}
 Industry: {industry}
  Skills: {skills_str}
 LinkedIn: https://linkedin.com{profile.get('linkedin_url', '')}
"""
        
        # Add highlights if available
        if result.highlight:
            output += "\n Highlights:\n"
            for field, highlights in result.highlight.items():
                for highlight in highlights[:2]:  # Show max 2 highlights per field
                    output += f"   • {highlight}\n"
        
        return output


def main():
    """Main interactive CLI"""
    print(" LinkedIn Profile Search Engine")
    print("="*50)
    
    # Initialize search engine
    try:
        engine = LinkedInSearchEngine()
    except Exception as e:
        print(f" Failed to connect to OpenSearch: {e}")
        print("Make sure OpenSearch is running on localhost:9200")
        return
    
    while True:
        print("\n" + "="*50)
        print("MENU:")
        print("1. Initialize (create index)")
        print("2. Load data from JSONL file")
        print("3. Search profiles")
        print("4. Get index stats")
        print("5. Exit")
        print("="*50)
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            # Initialize index
            try:
                engine.create_index()
            except Exception as e:
                print(f" Error creating index: {e}")
        
        elif choice == '2':
            # Load data
            file_path = input("Enter JSONL file path (or press Enter for 'data_001.jsonl'): ").strip()
            if not file_path:
                file_path = "data_001.jsonl"
            
            max_records_input = input("Max records to load (press Enter for all): ").strip()
            max_records = int(max_records_input) if max_records_input else None
            
            try:
                engine.load_jsonl_data(file_path, max_records)
            except Exception as e:
                print(f" Error loading data: {e}")
        
        elif choice == '3':
            # Search
            while True:
                print("\n" + "-"*40)
                print("SEARCH MODE")
                print("-"*40)
                
                query = input("\n Enter your search query (or 'back' to return to menu): ").strip()
                
                if query.lower() == 'back':
                    break
                
                if not query:
                    print(" Please enter a search query")
                    continue
                
                # Search method
                print("\nSearch methods:")
                print("1. Hybrid (recommended)")
                print("2. Keyword search")
                print("3. Semantic search")
                
                method_choice = input("Select method (1-3, default=1): ").strip()
                method_map = {'1': 'hybrid', '2': 'keyword', '3': 'semantic'}
                method = method_map.get(method_choice, 'hybrid')
                
                # Number of results
                size_input = input("Number of results (default=5): ").strip()
                size = int(size_input) if size_input else 5
                
                print(f"\n Searching for: '{query}' using {method} method...")
                print(" Please wait...\n")
                
                try:
                    # Parse query to show understanding
                    parsed = engine.parse_natural_language_query(query)
                    print(" Query Understanding:")
                    if parsed['skills']:
                        print(f"   Skills: {', '.join(parsed['skills'])}")
                    if parsed['location']:
                        print(f"   Location: {', '.join(parsed['location'])}")
                    if parsed['job_titles']:
                        print(f"   Job Titles: {', '.join(parsed['job_titles'])}")
                    if parsed['experience_level']:
                        print(f"   Experience Level: {parsed['experience_level']}")
                    print()
                    
                    # Perform search
                    results = engine.search(query, method=method, size=size)
                    
                    if not results:
                        print(" No results found. Try different keywords or search method.")
                        continue
                    
                    print(f" Found {len(results)} results:")
                    
                    # Display results
                    for i, result in enumerate(results):
                        print(engine.format_result(result, i))
                    
                except Exception as e:
                    print(f" Search error: {e}")
        
        elif choice == '4':
            # Index stats
            try:
                count_response = engine.client.count(index=engine.index_name)
                stats_response = engine.client.indices.stats(index=engine.index_name)
                
                print(f"\n Index Statistics:")
                print(f"   Total documents: {count_response['count']}")
                print(f"   Index size: {stats_response['indices'][engine.index_name]['total']['store']['size_in_bytes'] / (1024*1024):.2f} MB")
                
            except Exception as e:
                print(f" Error getting stats: {e}")
        
        elif choice == '5':
            print(" Goodbye!")
            break
        
        else:
            print(" Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    main()
