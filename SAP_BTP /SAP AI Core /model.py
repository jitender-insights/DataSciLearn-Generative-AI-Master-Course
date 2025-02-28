def evaluate_duplicate_detection(predictions, ground_truth):
    """
    Simple evaluation function for duplicate detection system.
    
    Args:
        predictions: List of dictionaries with prediction results 
                    {'is_duplicate': bool, 'original_ticket_id': str, 'confidence': float}
        ground_truth: List of dictionaries with actual labels
                    {'is_duplicate': bool, 'original_ticket_id': str}
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Initialize counters
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # For ROC curve
    y_true = []
    scores = []
    
    # For ticket matching accuracy
    correct_ticket_matches = 0
    total_actual_duplicates = 0
    
    # Process each prediction
    for pred, truth in zip(predictions, ground_truth):
        y_true.append(1 if truth['is_duplicate'] else 0)
        scores.append(pred['confidence'])
        
        # Count for confusion matrix
        if truth['is_duplicate']:
            total_actual_duplicates += 1
            if pred['is_duplicate']:
                true_positives += 1
                # Check if the original ticket matches
                if pred['original_ticket_id'] == truth['original_ticket_id']:
                    correct_ticket_matches += 1
            else:
                false_negatives += 1
        else:
            if pred['is_duplicate']:
                false_positives += 1
            else:
                true_negatives += 1
    
    # Calculate derived metrics
    total = true_positives + false_positives + true_negatives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    ticket_match_accuracy = correct_ticket_matches / total_actual_duplicates if total_actual_duplicates > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "total_samples": total,
        "ticket_match_accuracy": ticket_match_accuracy,
        "y_true": y_true,
        "scores": scores
    }

def evaluate_by_threshold(predictions, ground_truth, thresholds=None):
    """
    Evaluate system at different confidence thresholds.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        thresholds: List of thresholds to evaluate
        
    Returns:
        List of dictionaries with metrics at each threshold
    """
    if thresholds is None:
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = []
    
    for threshold in thresholds:
        # Apply threshold to predictions
        thresholded_preds = []
        for pred in predictions:
            # Create a new prediction with threshold applied
            new_pred = pred.copy()
            new_pred['is_duplicate'] = pred['confidence'] >= threshold
            thresholded_preds.append(new_pred)
        
        # Evaluate with this threshold
        metrics = evaluate_duplicate_detection(thresholded_preds, ground_truth)
        metrics['threshold'] = threshold
        results.append(metrics)
    
    return results

def evaluate_by_component(predictions, ground_truth, tickets):
    """
    Evaluate performance by component.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        tickets: List of original ticket data with component info
        
    Returns:
        Dictionary mapping components to their metrics
    """
    # Group by component
    components = {}
    
    for i, ticket in enumerate(tickets):
        component = ticket['component']
        if component not in components:
            components[component] = {
                'preds': [],
                'truth': []
            }
        
        components[component]['preds'].append(predictions[i])
        components[component]['truth'].append(ground_truth[i])
    
    # Calculate metrics for each component
    results = {}
    for component, data in components.items():
        if len(data['preds']) >= 5:  # Only evaluate components with enough samples
            metrics = evaluate_duplicate_detection(data['preds'], data['truth'])
            metrics['sample_count'] = len(data['preds'])
            results[component] = metrics
    
    return results

def collect_performance_metrics(test_data, master_df, check_duplicate_func):
    """
    Process test data and collect performance metrics.
    
    Args:
        test_data: DataFrame with test tickets and ground truth labels
        master_df: DataFrame with historical tickets
        check_duplicate_func: Function to check for duplicates
        
    Returns:
        Dictionary with all evaluation metrics
    """
    predictions = []
    ground_truth = []
    tickets = []
    
    # Process each test case
    for _, row in test_data.iterrows():
        # Extract ticket data
        ticket = {
            'ticket_id': row['ticket_id'],
            'company_code': row['company_code'],
            'component': row['component'],
            'summary': row['summary'],
            'description': row['description']
        }
        
        # Get prediction
        start_time = time.time()
        prediction = check_duplicate_func(master_df, ticket)
        end_time = time.time()
        prediction['processing_time'] = end_time - start_time
        
        # Store prediction
        predictions.append(prediction)
        
        # Store ground truth
        truth = {
            'is_duplicate': bool(row['is_duplicate']),  # Ensure boolean
            'original_ticket_id': row['original_ticket_id'] if row['is_duplicate'] else None
        }
        ground_truth.append(truth)
        tickets.append(ticket)
    
    # Calculate overall metrics
    overall_metrics = evaluate_duplicate_detection(predictions, ground_truth)
    
    # Calculate threshold-based metrics
    threshold_metrics = evaluate_by_threshold(predictions, ground_truth)
    
    # Calculate component-based metrics
    component_metrics = evaluate_by_component(predictions, ground_truth, tickets)
    
    # Calculate timing metrics
    processing_times = [p['processing_time'] for p in predictions]
    timing_metrics = {
        'mean_time': sum(processing_times) / len(processing_times),
        'max_time': max(processing_times),
        'min_time': min(processing_times),
        'total_time': sum(processing_times)
    }
    
    return {
        'overall': overall_metrics,
        'by_threshold': threshold_metrics,
        'by_component': component_metrics,
        'timing': timing_metrics
    }

# Function to prepare test data with labels from a CSV file
def prepare_test_data(test_csv_path):
    """
    Load and prepare test data from CSV.
    
    The CSV should contain:
    - ticket_id: ID of the test ticket
    - company_code: Company code
    - component: Ticket component
    - summary: Ticket summary
    - description: Ticket description
    - is_duplicate: Boolean flag (1/0) indicating if ticket is a duplicate
    - original_ticket_id: ID of the original ticket if is_duplicate=1
    
    Returns:
        DataFrame with test data
    """
    import pandas as pd
    
    # Load CSV
    test_data = pd.read_csv(test_csv_path)
    
    # Ensure required columns exist
    required_cols = ['ticket_id', 'company_code', 'component', 'summary', 
                     'description', 'is_duplicate', 'original_ticket_id']
    
    for col in required_cols:
        if col not in test_data.columns:
            if col == 'original_ticket_id':
                # Create empty column for original_ticket_id if missing
                test_data['original_ticket_id'] = None
            else:
                raise ValueError(f"Required column '{col}' not found in test data")
    
    # Ensure data types
    test_data['is_duplicate'] = test_data['is_duplicate'].astype(bool)
    
    # Ensure original_ticket_id is None for non-duplicates
    test_data.loc[~test_data['is_duplicate'], 'original_ticket_id'] = None
    
    return test_data
