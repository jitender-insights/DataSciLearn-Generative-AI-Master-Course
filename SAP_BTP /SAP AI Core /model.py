def get_vector_metrics():
    """Return vector search performance metrics"""
    vector_data = pd.DataFrame(metrics.metrics['vector_searches'])
    threshold_data = pd.DataFrame(metrics.metrics['threshold_checks'])
    
    # Count threshold crossings correctly
    threshold_counts = {
        'definite': 0,
        'likely': 0,
        'below_threshold': 0
    }
    
    if not threshold_data.empty:
        threshold_counts = {
            'definite': len(threshold_data[threshold_data['threshold_type'] == 'definite']),
            'likely': len(threshold_data[threshold_data['threshold_type'] == 'likely']),
            'below_threshold': len(threshold_data[threshold_data['threshold_type'] == 'below_threshold'])
        }
    
    return {
        'avg_search_time': vector_data['search_time'].mean() if not vector_data.empty else 0,
        'successful_thresholds': threshold_counts,
        'similarity_distribution': {
            'duplicates': [s['score'] for s in metrics.metrics['similarities'] if s['is_duplicate']],
            'non_duplicates': [s['score'] for s in metrics.metrics['similarities'] if not s['is_duplicate']]
        }
    }
