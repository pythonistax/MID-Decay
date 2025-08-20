import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import plot_tree, export_text
import numpy as np

# Simple Random Forest visualization that works with your current setup
print("Creating Random Forest visualizations...")

# Make sure you have the required variables
if 'best_model' not in locals():
    print("Error: 'best_model' not found. Please run your Random Forest training code first.")
else:
    rf_model = best_model.named_steps['model']
    
    # Check if it's actually a Random Forest
    if hasattr(rf_model, 'estimators_'):
        print(f"✓ Random Forest model detected with {rf_model.n_estimators} trees")
        
        # 1) Feature Importance Plot
        plt.figure(figsize=(12, 8))
        feature_importance = rf_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(15)
        
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title('Top 15 Feature Importances - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        # 2) Show first tree (simplified view)
        plt.figure(figsize=(25, 15))
        plot_tree(rf_model.estimators_[0], 
                 feature_names=X_train.columns,
                 filled=True,
                 rounded=True,
                 fontsize=10,
                 max_depth=3)  # Limit depth for readability
        plt.title('Random Forest - First Tree (Depth Limited to 3)')
        plt.tight_layout()
        plt.show()
        
        # 3) Text representation of decision rules
        print("\\n" + "="*60)
        print("DECISION RULES FROM FIRST TREE (First 1500 characters):")
        print("="*60)
        tree_text = export_text(rf_model.estimators_[0], 
                               feature_names=list(X_train.columns),
                               max_depth=3)
        print(tree_text[:1500] + "..." if len(tree_text) > 1500 else tree_text)
        
        # 4) Model statistics
        tree_depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
        tree_leaves = [tree.tree_.n_leaves for tree in rf_model.estimators_]
        
        print("\\n" + "="*60)
        print("RANDOM FOREST STATISTICS:")
        print("="*60)
        print(f"Number of trees: {rf_model.n_estimators}")
        print(f"Average tree depth: {np.mean(tree_depths):.2f}")
        print(f"Max tree depth: {max(tree_depths)}")
        print(f"Average number of leaves: {np.mean(tree_leaves):.2f}")
        print(f"Features with importance > 0: {sum(feature_importance > 0)}")
        
        # 5) Top business insights
        print("\\n" + "="*60)
        print("KEY BUSINESS INSIGHTS:")
        print("="*60)
        top_5_features = importance_df.head(5)
        for i, row in top_5_features.iterrows():
            print(f"{row.name + 1}. {row['feature']}: {row['importance']:.1%} importance")
        
    else:
        print(f"Error: Expected Random Forest, got {type(rf_model).__name__}")
        print("This model type doesn't have individual trees to visualize.")