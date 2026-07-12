import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure MLflow to track locally
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Hotel_Booking_Base_Model")

# Enable MLflow autologging for scikit-learn
mlflow.sklearn.autolog()

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=["is_canceled"])
    y_train = train_df["is_canceled"]
    
    X_test = test_df.drop(columns=["is_canceled"])
    y_test = test_df["is_canceled"]
    
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data(
        "hotel_bookings_preprocessing/train.csv",
        "hotel_bookings_preprocessing/test.csv"
    )

    with mlflow.start_run(run_name="RandomForest_Base"):
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate basic metrics (though autolog handles a lot)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
