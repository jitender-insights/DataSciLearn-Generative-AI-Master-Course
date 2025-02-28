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
            'threshold_checks': [],
            'decision_times': [],
            'parse_times': []
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
        
    def log_decision_time(self, duration):
        self.metrics['decision_times'].append(duration)
        
    def log_parse_attempt_time(self, duration):
        self.metrics['parse_times'].append(duration)
