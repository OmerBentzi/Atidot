"""
Main entry point for the Decision Assistant pipeline.
"""
import os
import sys
import time
import json
import pickle
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import set_seeds
from data_generator import make_synthetic_panel, temporal_split, validate_temporal_integrity
from modeling import prepare_features, train_xgb_with_search, evaluate_model, shap_global_bar
from rag import generate_lapse_plans, generate_lead_plans


def setup_logging(out_dir: str) -> None:
    """Setup logging to file and console."""
    log_path = os.path.join(out_dir, 'logs.txt')
    
    # Configure file handler with UTF-8 encoding for unicode paths
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )


def main():
    """Main pipeline execution."""
    start_time = time.time()
    
    # Setup
    out_dir = 'out'
    os.makedirs(out_dir, exist_ok=True)
    
    setup_logging(out_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Decision Assistant Pipeline Starting")
    logger.info("=" * 60)
    
    # Set seeds for reproducibility
    set_seeds(seed=42)
    logger.info(f"Seeds set for reproducibility (seed=42, PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', 'not set')})")
    
    # 1. Data generation
    phase_start = time.time()
    logger.info("\n[1/10] Generating synthetic panel data...")
    df = make_synthetic_panel(n_policies=2000, months=12, seed=42)
    logger.info(f"Generated {len(df)} rows across {df['month'].nunique()} months")
    logger.info(f"  Time: {time.time() - phase_start:.2f}s")
    
    # 2. Temporal split
    phase_start = time.time()
    logger.info("\n[2/10] Performing temporal split...")
    split = temporal_split(df, train_months=7, val_months=2)
    logger.info(f"Train: {len(split['train'])} rows, "
                f"Val: {len(split['val'])} rows, "
                f"Test: {len(split['test'])} rows")
    logger.info(f"  Time: {time.time() - phase_start:.2f}s")
    
    # 3. Validate temporal integrity
    phase_start = time.time()
    logger.info("\n[3/10] Validating temporal integrity...")
    validate_temporal_integrity(split)
    logger.info("[OK] Temporal integrity validated")
    logger.info(f"  Time: {time.time() - phase_start:.2f}s")
    
    # 4. Feature preparation
    phase_start = time.time()
    logger.info("\n[4/10] Preparing features...")
    feature_data = prepare_features(split, exclude_leakage=True)
    
    # Leakage guard assertion
    leakage_features = [f for f in feature_data['feature_names'] if f.startswith('post_')]
    assert len(leakage_features) == 0, f"Leakage detected: {leakage_features}"
    logger.info(f"[OK] Prepared {len(feature_data['feature_names'])} features (leakage excluded)")
    logger.info(f"  Time: {time.time() - phase_start:.2f}s")
    
    # 5. Train model
    phase_start = time.time()
    logger.info("\n[5/10] Training XGBoost with hyperparameter search...")
    model, best_params = train_xgb_with_search(
        feature_data['X_train'],
        feature_data['y_train'],
        feature_data['X_val'],
        feature_data['y_val'],
        random_state=42,
        max_trials=30
    )
    logger.info(f"[OK] Training complete. Best params: {best_params}")
    logger.info(f"  Time: {time.time() - phase_start:.2f}s")
    
    # Evaluate on validation set for logging
    val_pred_proba = model.predict_proba(feature_data['X_val'])[:, 1]
    from utils import auc_pr
    val_auc_pr = auc_pr(feature_data['y_val'], val_pred_proba)
    logger.info(f"Validation AUC-PR: {val_auc_pr:.4f}")
    
    # 6. Evaluate on test set
    phase_start = time.time()
    logger.info("\n[6/10] Evaluating model on test set...")
    metrics = evaluate_model(
        model,
        feature_data['X_test'],
        feature_data['y_test'],
        k_list=[0.01, 0.05]
    )
    logger.info(f"  Time: {time.time() - phase_start:.2f}s")
    
    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[OK] Metrics saved to {metrics_path}")
    logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    logger.info(f"  Precision@1%: {metrics['precision@1%']:.4f}")
    logger.info(f"  Precision@5%: {metrics['precision@5%']:.4f}")
    
    # 7. SHAP global bar plot
    phase_start = time.time()
    logger.info("\n[7/10] Generating SHAP global bar plot...")
    # Use a sample of validation data for SHAP (faster)
    n_shap_samples = min(500, len(feature_data['X_val']))
    shap_indices = range(n_shap_samples)
    X_val_sample = feature_data['X_val'][shap_indices]
    
    shap_path = os.path.join(out_dir, 'shap_global_bar.png')
    shap_global_bar(
        model,
        X_val_sample,
        feature_data['feature_names'],
        shap_path
    )
    logger.info(f"  Time: {time.time() - phase_start:.2f}s")
    
    # 8. Lapse prevention plans
    phase_start = time.time()
    logger.info("\n[8/10] Generating lapse prevention plans...")
    
    # Get test predictions for selecting customers
    test_proba = model.predict_proba(feature_data['X_test'])[:, 1]
    test_df = split['test'].copy()
    test_df['predicted_prob'] = test_proba
    
    # Select 3 customers: lowest, median, highest risk
    sorted_test = test_df.sort_values('predicted_prob').reset_index(drop=True)
    n_test = len(sorted_test)
    
    customers = [
        {
            'policy_id': int(sorted_test.iloc[0]['policy_id']),
            'probability': float(sorted_test.iloc[0]['predicted_prob'])
        },
        {
            'policy_id': int(sorted_test.iloc[n_test // 2]['policy_id']),
            'probability': float(sorted_test.iloc[n_test // 2]['predicted_prob'])
        },
        {
            'policy_id': int(sorted_test.iloc[-1]['policy_id']),
            'probability': float(sorted_test.iloc[-1]['predicted_prob'])
        }
    ]
    
    lapse_corpus_dir = os.path.join('src', 'corpora', 'lapse')
    lapse_plans = generate_lapse_plans(lapse_corpus_dir, customers)
    
    lapse_plans_path = os.path.join(out_dir, 'lapse_plans.json')
    with open(lapse_plans_path, 'w') as f:
        json.dump(lapse_plans, f, indent=2)
    logger.info(f"[OK] Lapse plans saved to {lapse_plans_path}")
    logger.info(f"  Time: {time.time() - phase_start:.2f}s")
    
    # 9. Lead conversion plans
    phase_start = time.time()
    logger.info("\n[9/10] Generating lead conversion plans...")
    
    leads = [
        {
            'lead_id': 'lead_001',
            'age': 32,
            'region': 'North',
            'channel': 'online',
            'needs': 'family coverage',
            'objections': 'price concerns'
        },
        {
            'lead_id': 'lead_002',
            'age': 55,
            'region': 'Center',
            'channel': 'phone',
            'needs': 'comprehensive protection',
            'objections': 'timing not right'
        },
        {
            'lead_id': 'lead_003',
            'age': 28,
            'region': 'South',
            'channel': 'email',
            'needs': 'basic coverage',
            'objections': 'need validation'
        }
    ]
    
    lead_corpus_dir = os.path.join('src', 'corpora', 'lead')
    lead_plans = generate_lead_plans(lead_corpus_dir, leads)
    
    lead_plans_path = os.path.join(out_dir, 'lead_plans.json')
    with open(lead_plans_path, 'w') as f:
        json.dump(lead_plans, f, indent=2)
    logger.info(f"[OK] Lead plans saved to {lead_plans_path}")
    logger.info(f"  Time: {time.time() - phase_start:.2f}s")
    
    # 10. Save model
    phase_start = time.time()
    logger.info("\n[10/10] Saving model...")
    model_data = {
        'model': model,
        'preprocessor': feature_data['preprocessor'],
        'feature_names': feature_data['feature_names'],
        'best_params': best_params
    }
    
    model_path = os.path.join(out_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"[OK] Model saved to {model_path}")
    logger.info(f"  Time: {time.time() - phase_start:.2f}s")
    
    # Final summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Total runtime: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    logger.info(f"Output directory: {os.path.abspath(out_dir)}")
    logger.info("=" * 60)
    
    print(f"\n[OK] Done! Outputs saved to: {os.path.abspath(out_dir)}")


if __name__ == '__main__':
    main()
