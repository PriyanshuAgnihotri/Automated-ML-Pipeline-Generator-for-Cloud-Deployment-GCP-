from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

def validate_model(model, X_test, y_test, metrics, thresholds):
    """Validate model against test data and thresholds"""
    try:
        predictions = model.predict(X_test)
        
        results = {}
        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = accuracy_score(y_test, predictions)
            elif metric == 'precision':
                results[metric] = precision_score(y_test, predictions, average='weighted')
            elif metric == 'recall':
                results[metric] = recall_score(y_test, predictions, average='weighted')
            elif metric == 'f1_score':
                results[metric] = f1_score(y_test, predictions, average='weighted')
        
        # Check if all metrics meet thresholds
        validation_passed = True
        for metric, threshold in thresholds.items():
            if results[metric] < threshold:
                logging.warning(f"Metric {metric} ({results[metric]}) below threshold ({threshold})")
                validation_passed = False
        
        logging.info("Model validation completed")
        return validation_passed, results
    except Exception as e:
        logging.error(f"Error in model validation: {e}")
        raise
    