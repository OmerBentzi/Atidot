"""
Model training, evaluation, and explainability.
"""
import numpy as np
import pandas as pd
import pickle
import json
import logging
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score
import shap
import matplotlib.pyplot as plt
from utils import month_to_ordinal, auc_pr, precision_at_k


logger = logging.getLogger(__name__)


def prepare_features(split: Dict[str, pd.DataFrame], exclude_leakage: bool = True) -> Dict[str, Any]:
    """
    Prepare features for modeling: encode categoricals, add month_ord, exclude leakage.
    
    Args:
        split: Dictionary with 'train', 'val', 'test' DataFrames
        exclude_leakage: If True, exclude columns starting with 'post_'
    
    Returns:
        Dictionary with:
        - 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test' (numpy arrays)
        - 'feature_names' (list)
        - 'preprocessor' (sklearn transformer)
    """
    train_df = split['train'].copy()
    val_df = split['val'].copy()
    test_df = split['test'].copy()
    
    # Add month_ord
    for df in [train_df, val_df, test_df]:
        df['month_ord'] = df['month'].apply(month_to_ordinal)
    
    # Define feature columns (including engineered features)
    numeric_features = ['age', 'tenure_m', 'premium', 'coverage', 'dependents', 'month_ord']
    # Add engineered features if they exist
    if 'premium_change_rate' in train_df.columns:
        numeric_features.append('premium_change_rate')
    if 'region_risk' in train_df.columns:
        numeric_features.append('region_risk')
    
    categorical_features = ['region']
    binary_features = ['has_agent', 'is_smoker']
    
    all_features = numeric_features + categorical_features + binary_features
    
    # Leakage guard: exclude any columns starting with 'post_'
    if exclude_leakage:
        leakage_cols = [col for col in train_df.columns if col.startswith('post_')]
        if leakage_cols:
            logger.warning(f"Excluding leakage columns: {leakage_cols}")
            assert all(col not in all_features for col in leakage_cols), \
                "Leakage columns found in feature list!"
    
    # Extract X and y
    X_train = train_df[all_features].copy()
    y_train = train_df['lapse_next_3m'].values
    X_val = val_df[all_features].copy()
    y_val = val_df['lapse_next_3m'].values
    X_test = test_df[all_features].copy()
    y_test = test_df['lapse_next_3m'].values
    
    # Build preprocessing pipeline
    # OHE for categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop'
    )
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    feature_names = []
    
    # Categorical OHE names
    cat_encoder = preprocessor.named_transformers_['cat']
    for i, cat_feat in enumerate(categorical_features):
        categories = cat_encoder.categories_[i]
        if len(categories) > 1:
            # Drop first category
            for cat_val in categories[1:]:
                feature_names.append(f"{cat_feat}_{cat_val}")
    
    # Numeric and binary features
    feature_names.extend(numeric_features)
    feature_names.extend(binary_features)
    
    # Final leakage guard: assert no post_* in feature names
    leakage_in_features = [name for name in feature_names if name.startswith('post_')]
    assert len(leakage_in_features) == 0, \
        f"Leakage detected in feature names: {leakage_in_features}"
    
    return {
        'X_train': X_train_processed,
        'y_train': y_train,
        'X_val': X_val_processed,
        'y_val': y_val,
        'X_test': X_test_processed,
        'y_test': y_test,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }


def train_xgb_with_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
    max_trials: int = 30
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Train XGBoost with randomized search and early stopping.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        random_state: Random seed
        max_trials: Maximum number of hyperparameter trials
    
    Returns:
        Tuple of (best_model, best_params)
    """
    # Expanded parameter grid for RandomizedSearchCV (~30 trials)
    param_distributions = {
        'max_depth': [4, 5, 6, 7],  # Allow deeper trees
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.01, 0.05, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2, 2.5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'learning_rate': [0.03, 0.05, 0.08, 0.1],  # Lower learning rates
        'n_estimators': [100, 150, 200, 250]
    }
    
    # Base estimator
    base_estimator = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=random_state,
        n_jobs=1,
        tree_method='hist'  # Fast and deterministic
    )
    
    # RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_distributions,
        n_iter=max_trials,
        cv=3,  # Will use train/val split manually for early stopping
        scoring='average_precision',
        n_jobs=1,
        random_state=random_state,
        verbose=1
    )
    
    # Compute baseline precision for comparison
    baseline_proba = np.full_like(y_val, y_train.mean(), dtype=float)
    baseline_aucpr = auc_pr(y_val, baseline_proba)
    logger.info(f"Baseline AUC-PR (constant prediction): {baseline_aucpr:.4f}")
    
    # For early stopping, we need to fit with eval_set
    # So we'll do a simplified search: fit models with different params and use early stopping
    best_score = -np.inf
    best_model = None
    best_params = None
    
    np.random.seed(random_state)
    
    for trial in range(max_trials):
        # Sample parameters
        params = {
            'max_depth': np.random.choice(param_distributions['max_depth']),
            'min_child_weight': np.random.choice(param_distributions['min_child_weight']),
            'gamma': np.random.choice(param_distributions['gamma']),
            'reg_alpha': np.random.choice(param_distributions['reg_alpha']),
            'reg_lambda': np.random.choice(param_distributions['reg_lambda']),
            'subsample': np.random.choice(param_distributions['subsample']),
            'colsample_bytree': np.random.choice(param_distributions['colsample_bytree']),
            'learning_rate': np.random.choice(param_distributions['learning_rate']),
            'n_estimators': np.random.choice(param_distributions['n_estimators'])
        }
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=random_state,
            n_jobs=1,
            tree_method='hist',
            **params
        )
        
        # Train with early stopping on validation set
        # Use callbacks for newer XGBoost versions (2.0+)
        try:
            # Try newer API with callbacks
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[xgb.callback.EarlyStopping(rounds=10, save_best=True)],
                verbose=False
            )
        except (TypeError, AttributeError):
            # Fallback to older API (pre-2.0)
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            except TypeError:
                # If neither works, fit without early stopping
                model.fit(X_train, y_train, verbose=False)
        
        # Evaluate on validation set
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_score = auc_pr(y_val, y_val_pred_proba)
        
        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_params = params.copy()
            logger.info(f"Trial {trial + 1}/{max_trials}: New best val AUC-PR = {best_score:.4f}")
    
    logger.info(f"Best validation AUC-PR: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    return best_model, best_params


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_list: List[float] = [0.01, 0.05]
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        k_list: List of k ratios for precision@k
    
    Returns:
        Dictionary with metrics: auc_pr, precision@1%, precision@5%, etc.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results = {
        'auc_pr': auc_pr(y_test, y_pred_proba)
    }
    
    for k_ratio in k_list:
        k_pct = int(k_ratio * 100)
        results[f'precision@{k_pct}%'] = precision_at_k(y_test, y_pred_proba, k_ratio)
    
    return results


def shap_global_bar(
    model: xgb.XGBClassifier,
    X_val_sample: np.ndarray,
    feature_names: List[str],
    out_path: str
) -> None:
    """
    Generate global SHAP bar plot for top features.
    Falls back to feature importance if SHAP fails.
    
    Args:
        model: Trained XGBoost model
        X_val_sample: Sample of validation data for SHAP computation
        feature_names: List of feature names
        out_path: Output path for PNG
    """
    use_shap = False
    mean_shap = None
    
    # Try SHAP - if anything fails, use feature importance
    try:
        explainer = shap.TreeExplainer(model)
        shap_output = explainer(X_val_sample)
        
        if hasattr(shap_output, 'values'):
            shap_values = shap_output.values
        else:
            shap_values = shap_output
            
        # Convert to numpy
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        shap_values = np.asarray(shap_values, dtype=np.float64)
        
        # Handle dimensions
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        elif shap_values.ndim > 2:
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 0]
            else:
                shap_values = shap_values.reshape(shap_values.shape[0], -1)
        
        # Compute mean absolute SHAP
        mean_shap = np.abs(shap_values).mean(axis=0)
        use_shap = True
        
    except Exception as e:
        logger.warning(f"SHAP failed ({type(e).__name__}: {e}), using feature importance instead")
        use_shap = False
    
    # Fallback to feature importance
    if mean_shap is None:
        mean_shap = model.feature_importances_
    
    # Get top 20 features
    top_k = min(20, len(feature_names))
    top_indices = np.argsort(mean_shap)[::-1][:top_k]
    
    top_features = [feature_names[i] for i in top_indices]
    top_shap = mean_shap[top_indices]
    
    # Plot
    try:
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_shap[::-1])
        plt.yticks(range(len(top_features)), top_features[::-1])
        plt.xlabel('Mean |SHAP value|' if use_shap else 'Feature Importance')
        plt.title('Top Features by Global SHAP Importance' if use_shap else 'Top Features by Importance')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to {out_path}")
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
