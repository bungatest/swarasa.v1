import os
import mlflow
import dagshub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=["is_canceled"])
    y_train = train_df["is_canceled"]
    
    X_test = test_df.drop(columns=["is_canceled"])
    y_test = test_df["is_canceled"]
    
    return X_train, X_test, y_train, y_test

def main():
    # Retrieve credentials from environment (set by user)
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "bungatest")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "my-first-repo")

    # Initialize DagsHub MLflow tracking
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    # Enable automatic logging for scikit-learn
    mlflow.sklearn.autolog()
    
    X_train, X_test, y_train, y_test = load_data(
        "hotel_bookings_preprocessing/train.csv",
        "hotel_bookings_preprocessing/test.csv"
    )

    # Use manual logging for Skilled/Advanced
    with mlflow.start_run(run_name="RandomForest_Tuned"):
        # Grid Search
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        
        # Log parameters explicitly with manual_ prefix to meet criteria
        mlflow.log_params({
            "manual_best_n_estimators": grid_search.best_params_['n_estimators'],
            "manual_best_max_depth": grid_search.best_params_['max_depth']
        })

        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics with test_ prefix to avoid clashing with autolog
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        # Artifact 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        
        # Artifact 2: Feature Importance
        feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
        plt.figure(figsize=(10, 8))
        feature_importances.nlargest(15).plot(kind='barh')
        plt.title('Top 15 Feature Importances')
        fi_path = "feature_importance.png"
        plt.savefig(fi_path)
        plt.close()
        mlflow.log_artifact(fi_path)

        # Log Model
        mlflow.sklearn.log_model(best_model, "model")

        print("Hyperparameter tuning and logging to DagsHub completed.")
        print(f"Best Params: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
