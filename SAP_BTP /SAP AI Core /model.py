# Added imports for metrics
import time
from collections import defaultdict
import pandas as pd

# ====================
# Metrics Collection
# ====================

class MetricsCollector:
    def __init__(self):
        self.reset_metrics()
        
    def reset_metrics(self):
        self.metrics = {
            'vector_searches': [],
            'llm_calls': [],
            'decisions': [],
            'similarities': [],
            'parse_attempts': [],
            'threshold_checks': []
        }
        
    def log_vector_search(self, k, num_candidates, search_time):
        self.metrics['vector_searches'].append({
            'timestamp': time.time(),
            'k': k,
            'candidates_available': num_candidates,
            'search_time': search_time
        })
        
    def log_llm_call(self, success, retries, response_time):
        self.metrics['llm_calls'].append({
            'timestamp': time.time(),
            'success': success,
            'retries': retries,
            'response_time': response_time
        })
        
    def log_decision(self, is_duplicate, confidence, method):
        self.metrics['decisions'].append({
            'timestamp': time.time(),
            'is_duplicate': is_duplicate,
            'confidence': confidence,
            'method': method
        })
        
    def log_similarity(self, score, is_duplicate):
        self.metrics['similarities'].append({
            'timestamp': time.time(),
            'score': score,
            'is_duplicate': is_duplicate
        })
        
    def log_parse_attempt(self, success):
        self.metrics['parse_attempts'].append({
            'timestamp': time.time(),
            'success': success
        })
        
    def log_threshold_check(self, threshold_type, score):
        self.metrics['threshold_checks'].append({
            'timestamp': time.time(),
            'threshold_type': threshold_type,
            'score': score
        })

# Initialize global metrics collector
metrics = MetricsCollector()

# ====================
# Modified Functions with Metrics
# ====================

def search_vector_db(index, query_embedding, k=5):
    """Search vector database for similar tickets with metrics"""
    start_time = time.time()
    
    # Original search logic
    query_embedding = query_embedding.reshape(1, -1)
    k = min(k, index.ntotal)
    
    if k == 0:
        metrics.log_vector_search(k, 0, 0)
        return np.array([]), np.array([])
        
    distances, indices = index.search(query_embedding, k)
    search_time = time.time() - start_time
    
    # Log metrics
    metrics.log_vector_search(
        k=k,
        num_candidates=index.ntotal,
        search_time=search_time
    )
    
    return distances[0], indices[0]

def parse_llm_response(response_text, default_response):
    """Parse LLM response with metrics"""
    parse_start = time.time()
    try:
        # Original parsing logic
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "")
            
        parsed = json.loads(response_text)
        if "is_duplicate" not in parsed and "is_global_outage" not in parsed:
            metrics.log_parse_attempt(False)
            return default_response
            
        metrics.log_parse_attempt(True)
        return parsed
    except Exception as e:
        metrics.log_parse_attempt(False)
        print(f"Error parsing LLM response: {str(e)}")
        return default_response
    finally:
        metrics.log_parse_attempt_time(time.time() - parse_start)

def check_duplicate(master_df, ticket_data):
    """Check duplicates with full metrics tracking"""
    start_time = time.time()
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "Unable to perform analysis"
    }

    try:
        # Original validation logic
        
        final_result = None
        method_used = 'none'
        
        if config.ENABLE_VECTOR_DB_DUPLICATE_DETECTION:
            # Vector DB logic
            relevant_df = master_df[...]  # Your existing code
            
            # After similarity calculation
            best_similarity = ...  # Your existing code
            final_result = ...    # Your existing decision
            
            if final_result['is_duplicate']:
                metrics.log_similarity(best_similarity, True)
                method_used = 'vector'
            else:
                metrics.log_similarity(best_similarity, False)
                
            metrics.log_threshold_check(
                'definite' if best_similarity >= config.DUPLICATE_THRESHOLDS['definite_duplicate'] 
                else 'likely' if best_similarity >= config.DUPLICATE_THRESHOLDS['likely_duplicate'] 
                else 'none',
                best_similarity
            )

        if not final_result and config.ENABLE_LLM_DUPLICATE_DETECTION:
            # LLM logic
            llm_start = time.time()
            retries = 0
            while retries < config.MAX_LLM_RETRIES:
                try:
                    response = chain.invoke(...)  # Your existing code
                    result = parse_llm_response(response, default_response)
                    
                    if result != default_response:
                        final_result = result
                        method_used = 'llm'
                        metrics.log_llm_call(
                            success=True,
                            retries=retries,
                            response_time=time.time() - llm_start
                        )
                        break
                        
                    retries += 1
                except Exception as e:
                    retries += 1
                    
            if not final_result:
                metrics.log_llm_call(
                    success=False,
                    retries=retries,
                    response_time=time.time() - llm_start
                )

        # Log final decision
        if final_result:
            metrics.log_decision(
                is_duplicate=final_result['is_duplicate'],
                confidence=final_result['confidence'],
                method=method_used
            )
            
        return final_result or default_response
        
    finally:
        metrics.log_decision_time(time.time() - start_time)

# ====================
# Metrics Analysis Functions
# ====================

def get_system_metrics():
    """Return metrics for system performance"""
    return {
        'total_tickets': len(metrics.metrics['decisions']),
        'duplicate_rate': (
            sum(1 for d in metrics.metrics['decisions'] if d['is_duplicate']) /
            len(metrics.metrics['decisions']) if metrics.metrics['decisions'] else 0
        ),
        'avg_confidence': (
            sum(d['confidence'] for d in metrics.metrics['decisions']) /
            len(metrics.metrics['decisions']) if metrics.metrics['decisions'] else 0
        )
    }

def get_vector_metrics():
    """Return vector search performance metrics"""
    vector_data = pd.DataFrame(metrics.metrics['vector_searches'])
    threshold_data = pd.DataFrame(metrics.metrics['threshold_checks'])
    
    return {
        'avg_search_time': vector_data['search_time'].mean() if not vector_data.empty else 0,
        'successful_thresholds': {
            'definite': len(threshold_data[threshold_data['threshold_type'] == 'definite']),
            'likely': len(threshold_data[threshold_data['threshold_type'] == 'likely']),
            'none': len(threshold_data[threshold_data['threshold_type'] == 'none'])
        },
        'similarity_distribution': {
            'duplicates': [s['score'] for s in metrics.metrics['similarities'] if s['is_duplicate']],
            'non_duplicates': [s['score'] for s in metrics.metrics['similarities'] if not s['is_duplicate']]
        }
    }

def get_llm_metrics():
    """Return LLM performance metrics"""
    llm_data = pd.DataFrame(metrics.metrics['llm_calls'])
    parse_data = pd.DataFrame(metrics.metrics['parse_attempts'])
    
    return {
        'success_rate': llm_data['success'].mean() if not llm_data.empty else 0,
        'avg_retries': llm_data['retries'].mean() if not llm_data.empty else 0,
        'parse_success_rate': parse_data['success'].mean() if not parse_data.empty else 0,
        'avg_response_time': llm_data['response_time'].mean() if not llm_data.empty else 0
    }

def get_all_metrics():
    """Return combined metrics for dashboard"""
    return {
        'system': get_system_metrics(),
        'vector': get_vector_metrics(),
        'llm': get_llm_metrics(),
        'raw_data': {
            'decisions': metrics.metrics['decisions'],
            'similarities': metrics.metrics['similarities']
        }
    }
