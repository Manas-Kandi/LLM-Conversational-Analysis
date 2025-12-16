#!/usr/bin/env python3
"""
Breakdown Predictor
Predicts which conversations will break down into gibberish based on early-turn features
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


class BreakdownPredictor:
    """Predicts conversation breakdown using early-turn features"""
    
    def __init__(self, turns_df: pd.DataFrame):
        self.turns_df = turns_df.copy()
        self.model = None
        self.feature_names = None
        self.breakdown_threshold = 0.3  # Gibberish score threshold for breakdown
        
    def label_breakdown_conversations(self) -> pd.DataFrame:
        """Label conversations that break down based on max gibberish score"""
        # Find maximum gibberish score for each conversation
        max_gibberish = self.turns_df.groupby('conversation_id')['gibberish_score'].max()
        
        # Label as breakdown if exceeds threshold
        breakdown_labels = (max_gibberish > self.breakdown_threshold).astype(int)
        
        # Add labels to turns dataframe
        self.turns_df['will_breakdown'] = self.turns_df['conversation_id'].map(breakdown_labels)
        
        print(f"Labeled {breakdown_labels.sum()} out of {len(breakdown_labels)} conversations as breakdown")
        return self.turns_df
    
    def extract_early_turn_features(self, n_turns: int = 5) -> pd.DataFrame:
        """Extract features from first n turns of each conversation"""
        features_list = []
        
        for conv_id in self.turns_df['conversation_id'].unique():
            conv_turns = self.turns_df[self.turns_df['conversation_id'] == conv_id]
            early_turns = conv_turns[conv_turns['turn_number'] <= n_turns]
            
            if len(early_turns) == 0:
                continue
            
            # Get the label (will be same for all turns)
            will_breakdown = early_turns['will_breakdown'].iloc[0]
            
            # Extract features from early turns
            feature_dict = {
                'conversation_id': conv_id,
                'will_breakdown': will_breakdown,
                'n_early_turns': len(early_turns)
            }
            
            # Agent A features
            agent_a_turns = early_turns[early_turns['role'] == 'agent_a']
            if len(agent_a_turns) > 0:
                for feature in ['word_count', 'sentiment_score', 'formality_score', 
                               'complexity_score', 'ai_reference_count', 'gibberish_score']:
                    feature_dict[f'agent_a_{feature}_mean'] = agent_a_turns[feature].mean()
                    feature_dict[f'agent_a_{feature}_std'] = agent_a_turns[feature].std()
                    feature_dict[f'agent_a_{feature}_trend'] = np.polyfit(
                        range(len(agent_a_turns)), agent_a_turns[feature].values, 1
                    )[0] if len(agent_a_turns) > 1 else 0
            
            # Agent B features
            agent_b_turns = early_turns[early_turns['role'] == 'agent_b']
            if len(agent_b_turns) > 0:
                for feature in ['word_count', 'sentiment_score', 'formality_score',
                               'complexity_score', 'ai_reference_count', 'gibberish_score']:
                    feature_dict[f'agent_b_{feature}_mean'] = agent_b_turns[feature].mean()
                    feature_dict[f'agent_b_{feature}_std'] = agent_b_turns[feature].std()
                    feature_dict[f'agent_b_{feature}_trend'] = np.polyfit(
                        range(len(agent_b_turns)), agent_b_turns[feature].values, 1
                    )[0] if len(agent_b_turns) > 1 else 0
            
            # Interaction features
            if len(agent_a_turns) > 0 and len(agent_b_turns) > 0:
                feature_dict['word_count_ratio'] = (
                    agent_a_turns['word_count'].mean() / agent_b_turns['word_count'].mean()
                )
                feature_dict['sentiment_correlation'] = np.corrcoef(
                    agent_a_turns['sentiment_score'], agent_b_turns['sentiment_score']
                )[0, 1] if len(agent_a_turns) > 1 and len(agent_b_turns) > 1 else 0
            
            # Temperature and model features
            metadata = early_turns.iloc[0]
            feature_dict['agent_a_temp'] = metadata.get('agent_a_temp', 0)
            feature_dict['agent_b_temp'] = metadata.get('agent_b_temp', 0)
            
            features_list.append(feature_dict)
        
        return pd.DataFrame(features_list)
    
    def train_predictor(self, features_df: pd.DataFrame, model_type: str = 'random_forest'):
        """Train a model to predict breakdown"""
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['conversation_id', 'will_breakdown']]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['will_breakdown']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Choose model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=42)
        elif model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, class_weight='balanced')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.feature_names = feature_cols
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test, y_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        
        results = {
            'classification_report': report,
            'auc_score': auc_score,
            'cv_mean_auc': cv_scores.mean(),
            'cv_std_auc': cv_scores.std(),
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_))
        }
        
        return results, X_test, y_test, y_proba
    
    def analyze_early_warning_signs(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify early warning signs of breakdown"""
        breakdown_convs = features_df[features_df['will_breakdown'] == 1]
        stable_convs = features_df[features_df['will_breakdown'] == 0]
        
        warning_signs = {}
        
        for feature in features_df.columns:
            if feature in ['conversation_id', 'will_breakdown', 'n_early_turns']:
                continue
            
            breakdown_mean = breakdown_convs[feature].mean()
            stable_mean = stable_convs[feature].mean()
            
            # Calculate effect size
            pooled_std = np.sqrt(
                (breakdown_convs[feature].var() * (len(breakdown_convs) - 1) +
                 stable_convs[feature].var() * (len(stable_convs) - 1)) /
                (len(breakdown_convs) + len(stable_convs) - 2)
            )
            
            effect_size = (breakdown_mean - stable_mean) / (pooled_std + 1e-8)
            
            warning_signs[feature] = {
                'breakdown_mean': breakdown_mean,
                'stable_mean': stable_mean,
                'effect_size': effect_size,
                'warning_level': 'high' if abs(effect_size) > 0.8 else 'medium' if abs(effect_size) > 0.5 else 'low'
            }
        
        # Sort by effect size
        warning_signs = dict(sorted(warning_signs.items(), 
                                  key=lambda x: abs(x[1]['effect_size']), reverse=True))
        
        return warning_signs
    
    def visualize_predictions(self, X_test: pd.DataFrame, y_test: np.ndarray, 
                            y_proba: np.ndarray, output_dir: str):
        """Create visualizations for prediction results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True).tail(20)
            
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title('Top 20 Feature Importance for Breakdown Prediction')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for Breakdown Prediction')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Prediction confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_proba[y_test == 0], bins=30, alpha=0.7, label='Stable', density=True)
        plt.hist(y_proba[y_test == 1], bins=30, alpha=0.7, label='Breakdown', density=True)
        plt.xlabel('Predicted Probability of Breakdown')
        plt.ylabel('Density')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'prediction_confidence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")


def main():
    """Example usage"""
    # Load turn features
    turns_df = pd.read_csv("analysis_outputs/advanced_insights/turns_features.csv")
    
    # Initialize predictor
    predictor = BreakdownPredictor(turns_df)
    
    # Label breakdown conversations
    labeled_df = predictor.label_breakdown_conversations()
    
    # Extract early-turn features
    print("Extracting early-turn features...")
    features_df = predictor.extract_early_turn_features(n_turns=5)
    
    # Train predictor
    print("Training breakdown predictor...")
    results, X_test, y_test, y_proba = predictor.train_predictor(features_df)
    
    # Analyze early warning signs
    print("Analyzing early warning signs...")
    warning_signs = predictor.analyze_early_warning_signs(features_df)
    
    # Create visualizations
    predictor.visualize_predictions(X_test, y_test, y_proba, "analysis_outputs/breakdown_prediction")
    
    # Save results
    output = {
        'model_performance': {
            'auc_score': results['auc_score'],
            'cv_mean_auc': results['cv_mean_auc'],
            'cv_std_auc': results['cv_std_auc']
        },
        'early_warning_signs': warning_signs,
        'feature_importance': results['feature_importance']
    }
    
    with open("analysis_outputs/breakdown_prediction/breakdown_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nBreakdown Predictor Results:")
    print(f"AUC Score: {results['auc_score']:.3f}")
    print(f"Cross-validated AUC: {results['cv_mean_auc']:.3f} ± {results['cv_std_auc']:.3f}")
    
    print("\nTop Early Warning Signs:")
    for feature, signs in list(warning_signs.items())[:5]:
        print(f"  {feature}: effect_size={signs['effect_size']:.2f} ({signs['warning_level']})")


if __name__ == "__main__":
    main()
