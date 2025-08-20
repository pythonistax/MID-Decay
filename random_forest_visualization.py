import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def visualize_random_forest(best_model, X, feature_names, max_trees=3):
    """
    Create various visualizations for Random Forest model
    """
    
    rf_model = best_model.named_steps['model']
    
    # Check if it's actually a Random Forest model
    if not hasattr(rf_model, 'estimators_'):
        print(f"Error: Expected RandomForestRegressor, got {type(rf_model).__name__}")
        print("This visualization is specifically for Random Forest models.")
        print("For XGBoost visualization, use a different approach.")
        return None
    
    # 1) Feature Importance Plot
    plt.figure(figsize=(12, 8))
    feature_importance = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(20)
    
    plt.subplot(2, 2, 1)
    sns.barplot(data=importance_df, y='feature', x='importance')
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    
    # 2) Individual Tree Visualization (first few trees)
    for i in range(min(max_trees, len(rf_model.estimators_))):
        plt.figure(figsize=(20, 12))
        plot_tree(rf_model.estimators_[i], 
                 feature_names=feature_names,
                 filled=True,
                 rounded=True,
                 fontsize=8,
                 max_depth=3)  # Limit depth for readability
        plt.title(f'Random Forest - Tree {i+1} (max_depth=3 for visualization)')
        plt.tight_layout()
        plt.show()
    
    # 3) Tree Depth Distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    tree_depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
    plt.hist(tree_depths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Tree Depths')
    plt.xlabel('Tree Depth')
    plt.ylabel('Frequency')
    
    # 4) Number of Leaves Distribution
    plt.subplot(1, 2, 2)
    tree_leaves = [tree.tree_.n_leaves for tree in rf_model.estimators_]
    plt.hist(tree_leaves, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Number of Leaves')
    plt.xlabel('Number of Leaves')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # 5) Feature Importance Heatmap (grouped)
    plt.figure(figsize=(14, 8))
    
    # Group features by type
    rate_features = [f for f in feature_names if 'rate' in f.lower()]
    revenue_features = [f for f in feature_names if 'revenue' in f.lower()]
    charge_features = [f for f in feature_names if 'charge' in f.lower()]
    cycle_features = [f for f in feature_names if 'cycle' in f.lower()]
    other_features = [f for f in feature_names if f not in rate_features + revenue_features + charge_features + cycle_features]
    
    feature_groups = {
        'Rate Features': rate_features,
        'Revenue Features': revenue_features,
        'Charge Features': charge_features,
        'Cycle Features': cycle_features,
        'Other Features': other_features
    }
    
    group_importance = {}
    for group_name, features in feature_groups.items():
        group_features = [f for f in features if f in feature_names]
        if group_features:
            group_indices = [list(feature_names).index(f) for f in group_features]
            group_importance[group_name] = feature_importance[group_indices].sum()
    
    # Plot grouped importance
    groups = list(group_importance.keys())
    importances = list(group_importance.values())
    
    plt.pie(importances, labels=groups, autopct='%1.1f%%', startangle=90)
    plt.title('Feature Importance by Category')
    plt.axis('equal')
    plt.show()
    
    # 6) Text representation of first tree
    print("\\n" + "="*60)
    print("TEXT REPRESENTATION OF FIRST TREE:")
    print("="*60)
    tree_text = export_text(rf_model.estimators_[0], 
                           feature_names=list(feature_names),
                           max_depth=4)  # Limit depth for readability
    print(tree_text[:2000] + "..." if len(tree_text) > 2000 else tree_text)
    
    # 7) Model Statistics
    print("\\n" + "="*60)
    print("RANDOM FOREST MODEL STATISTICS:")
    print("="*60)
    print(f"Number of trees: {rf_model.n_estimators}")
    print(f"Average tree depth: {np.mean(tree_depths):.2f}")
    print(f"Max tree depth: {max(tree_depths)}")
    print(f"Min tree depth: {min(tree_depths)}")
    print(f"Average number of leaves: {np.mean(tree_leaves):.2f}")
    print(f"Total number of features: {len(feature_names)}")
    print(f"Features used (importance > 0): {sum(feature_importance > 0)}")
    
    return {
        'tree_depths': tree_depths,
        'tree_leaves': tree_leaves,
        'feature_importance': feature_importance,
        'group_importance': group_importance
    }

# Add this to your existing notebook cell or run separately:
print("Creating Random Forest visualizations...")

# Get the trained model and feature names from your existing code
# Make sure you have: best_model, X_train, and feature names

# Run the visualization
if 'best_model' in locals() and 'X_train' in locals():
    vis_results = visualize_random_forest(
        best_model, 
        X_train, 
        X_train.columns, 
        max_trees=2  # Show first 2 trees
    )
else:
    print("Please run your Random Forest training code first!")
    print("Make sure 'best_model' and 'X_train' variables are available.")