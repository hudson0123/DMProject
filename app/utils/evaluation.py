# app/utils/evaluation.py

from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from pathlib import Path

class ModelEvaluator:
    """
    Comprehensive model evaluation utilities for genre classification.
    Handles metric calculation, visualization, and error analysis.
    """
    
    def __init__(self, save_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            save_path: Directory to save evaluation artifacts
        """
        self.save_path = Path(save_path) if save_path else None
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing various metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
            "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Calculate per-class metrics
        unique_classes = np.unique(y_true)
        per_class_metrics = {}
        
        for cls in unique_classes:
            cls_mask = y_true == cls
            per_class_metrics[cls] = {
                "precision": precision_score(y_true == cls, y_pred == cls),
                "recall": recall_score(y_true == cls, y_pred == cls),
                "f1": f1_score(y_true == cls, y_pred == cls),
                "support": np.sum(cls_mask)
            }
            
            # Add ROC and PR curves if probabilities are available
            if y_prob is not None:
                cls_idx = list(unique_classes).index(cls)
                cls_prob = y_prob[:, cls_idx]
                
                # ROC curve
                fpr, tpr, _ = roc_curve(y_true == cls, cls_prob)
                per_class_metrics[cls]["roc_auc"] = auc(fpr, tpr)
                per_class_metrics[cls]["roc_curve"] = {"fpr": fpr, "tpr": tpr}
                
                # PR curve
                precision, recall, _ = precision_recall_curve(y_true == cls, cls_prob)
                per_class_metrics[cls]["avg_precision"] = average_precision_score(
                    y_true == cls, cls_prob
                )
                per_class_metrics[cls]["pr_curve"] = {
                    "precision": precision, "recall": recall
                }
        
        metrics["per_class"] = per_class_metrics
        
        return metrics
        
    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyze prediction errors to identify patterns.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            X: Feature matrix
            feature_names: Names of features
            
        Returns:
            Dictionary containing error analysis
        """
        error_mask = y_true != y_pred
        error_indices = np.where(error_mask)[0]
        
        error_analysis = {
            "total_errors": np.sum(error_mask),
            "error_rate": np.mean(error_mask),
            "error_distribution": {}
        }
        
        # Analyze misclassification patterns
        for true_label in np.unique(y_true):
            true_mask = y_true == true_label
            error_analysis["error_distribution"][true_label] = {
                "total": np.sum(true_mask & error_mask),
                "misclassified_as": dict(zip(
                    *np.unique(y_pred[true_mask & error_mask], return_counts=True)
                ))
            }
            
        # Analyze feature distributions for errors
        error_feature_stats = {}
        for i, feature in enumerate(feature_names):
            error_feature_stats[feature] = {
                "mean_error": np.mean(X[error_mask, i]),
                "std_error": np.std(X[error_mask, i]),
                "mean_correct": np.mean(X[~error_mask, i]),
                "std_correct": np.std(X[~error_mask, i])
            }
            
        error_analysis["feature_statistics"] = error_feature_stats
        
        return error_analysis
        
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, 
                            labels: List[str]) -> str:
        """
        Create and save confusion matrix visualization.
        
        Args:
            conf_matrix: Confusion matrix
            labels: Class labels
            
        Returns:
            Base64 encoded PNG image
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=300)
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        
        return img_str
        
    def plot_roc_curves(self, metrics: Dict[str, Any]) -> str:
        """
        Create and save ROC curves for all classes.
        
        Args:
            metrics: Metrics dictionary containing ROC curve data
            
        Returns:
            Base64 encoded PNG image
        """
        plt.figure(figsize=(10, 8))
        
        for class_name, class_metrics in metrics["per_class"].items():
            if "roc_curve" in class_metrics:
                fpr = class_metrics["roc_curve"]["fpr"]
                tpr = class_metrics["roc_curve"]["tpr"]
                roc_auc = class_metrics["roc_auc"]
                
                plt.plot(
                    fpr, tpr,
                    label=f'{class_name} (AUC = {roc_auc:.2f})'
                )
                
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        
        # Save plot
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=300)
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        
        return img_str
        
    def generate_report(self, metrics: Dict[str, Any], 
                       error_analysis: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive evaluation report in HTML format.
        
        Args:
            metrics: Model evaluation metrics
            error_analysis: Error analysis results (optional)
            
        Returns:
            HTML formatted report
        """
        html = """
        <div class="evaluation-report">
            <h2>Model Evaluation Report</h2>
            
            <div class="metrics-summary">
                <h3>Overall Metrics</h3>
                <ul>
                    <li>Accuracy: {:.3f}</li>
                    <li>Weighted Precision: {:.3f}</li>
                    <li>Weighted Recall: {:.3f}</li>
                    <li>Weighted F1: {:.3f}</li>
                    <li>Cohen's Kappa: {:.3f}</li>
                </ul>
            </div>
        """.format(
            metrics["accuracy"],
            metrics["precision_weighted"],
            metrics["recall_weighted"],
            metrics["f1_weighted"],
            metrics["cohen_kappa"]
        )
        
        # Add per-class metrics
        html += """
            <div class="per-class-metrics">
                <h3>Per-Class Performance</h3>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>Support</th>
                    </tr>
        """
        
        for cls, cls_metrics in metrics["per_class"].items():
            html += """
                <tr>
                    <td>{}</td>
                    <td>{:.3f}</td>
                    <td>{:.3f}</td>
                    <td>{:.3f}</td>
                    <td>{}</td>
                </tr>
            """.format(
                cls,
                cls_metrics["precision"],
                cls_metrics["recall"],
                cls_metrics["f1"],
                cls_metrics["support"]
            )
            
        html += "</table></div>"
        
        # Add error analysis if available
        if error_analysis:
            html += """
                <div class="error-analysis">
                    <h3>Error Analysis</h3>
                    <p>Total Errors: {}</p>
                    <p>Error Rate: {:.2%}</p>
                </div>
            """.format(
                error_analysis["total_errors"],
                error_analysis["error_rate"]
            )
            
        html += "</div>"
        
        return html

    def save_evaluation_artifacts(self, metrics: Dict[str, Any], 
                                error_analysis: Optional[Dict[str, Any]] = None) -> None:
        """
        Save all evaluation artifacts to disk.
        
        Args:
            metrics: Model evaluation metrics
            error_analysis: Error analysis results (optional)
        """
        if not self.save_path:
            return
            
        # Save metrics as JSON
        import json
        metrics_file = self.save_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Save visualizations
        if 'confusion_matrix' in metrics:
            conf_matrix_img = self.plot_confusion_matrix(
                metrics['confusion_matrix'],
                list(metrics['per_class'].keys())
            )
            with open(self.save_path / 'confusion_matrix.png', 'wb') as f:
                f.write(base64.b64decode(conf_matrix_img))
                
        # Save ROC curves
        roc_img = self.plot_roc_curves(metrics)
        with open(self.save_path / 'roc_curves.png', 'wb') as f:
            f.write(base64.b64decode(roc_img))
            
        # Save HTML report
        report = self.generate_report(metrics, error_analysis)
        with open(self.save_path / 'evaluation_report.html', 'w') as f:
            f.write(report)