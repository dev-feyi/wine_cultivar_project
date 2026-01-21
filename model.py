import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


class WineCultivarModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.cultivar_names = ['Cultivar 1', 'Cultivar 2', 'Cultivar 3']

    def load_data(self, file_path=None):
        """
        Load and preprocess wine data
        Returns: X (features), y (cultivar labels)
        """
        if file_path is None:
            # Use absolute path
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, 'data', 'wine.csv')

        try:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df)} wine samples")

            # Separate features and target
            X = df.drop('cultivar', axis=1)
            y = df['cultivar']

            # Store feature names
            self.feature_names = X.columns.tolist()

            print("\nDataset Statistics:")
            print(f"  Total Wine Samples: {len(df)}")
            for i in range(3):
                count = (y == i).sum()
                print(f"  Cultivar {i + 1}: {count} samples ({count / len(y) * 100:.1f}%)")
            print(f"  Number of Features: {len(self.feature_names)}")

            return X, y

        except FileNotFoundError:
            print(f"✗ Error: {file_path} not found")
            return None, None

    def train(self, X, y):
        """
        Train the wine cultivar prediction model
        Uses Random Forest Classifier
        """
        print("\n--- Training Model ---")

        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=5
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(f"\nModel Performance:")
        print(f"  Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"  Testing Accuracy: {test_accuracy * 100:.2f}%")

        # Detailed classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, test_predictions,
                                    target_names=self.cultivar_names))

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_predictions)
        print("\nConfusion Matrix:")
        print(cm)

        # Feature importance
        self._show_feature_importance()

        return test_accuracy

    def _show_feature_importance(self):
        """Display which features matter most for cultivar prediction"""
        if self.model and self.feature_names:
            importance = self.model.feature_importances_
            feature_imp = sorted(zip(self.feature_names, importance),
                                 key=lambda x: x[1], reverse=True)

            print("\nTop 10 Most Important Features:")
            for i, (name, imp) in enumerate(feature_imp[:10], 1):
                print(f"  {i}. {name}: {imp:.4f}")

    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Convert features dict to array if needed
        if isinstance(features, dict):
            features_array = np.array([[features[name] for name in self.feature_names]])
        else:
            features_array = np.array([features])

        # Scale features
        features_scaled = self.scaler.transform(features_array)

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        return prediction, probabilities

    def save_model(self, model_dir='model_files'):
        """Save trained model and scaler to disk"""
        # Use absolute path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, model_dir)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'wine_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        features_path = os.path.join(model_dir, 'feature_names.pkl')

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, features_path)

        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")
        print(f"✓ Feature names saved to {features_path}")

    def load_model(self, model_dir='model_files'):
        """Load pre-trained model and scaler from disk"""
        # Use absolute path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, model_dir)

        model_path = os.path.join(model_dir, 'wine_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        features_path = os.path.join(model_dir, 'feature_names.pkl')

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)
            print(f"✓ Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"✗ Model files not found in {model_dir}")
            return False


def train_and_save_model():
    """
    Main training function - run this to create the model
    """
    print("=" * 60)
    print("WINE CULTIVAR ORIGIN PREDICTION MODEL TRAINING")
    print("=" * 60)

    # Initialize model
    wine_model = WineCultivarModel()

    # Load data
    X, y = wine_model.load_data()
    if X is None:
        return

    # Train model
    wine_model.train(X, y)

    # Save trained model
    wine_model.save_model()

    # Test predictions
    print("\n--- Testing Sample Predictions ---")

    # Get samples from each cultivar
    for cultivar in [0, 1, 2]:
        sample = X.iloc[y[y == cultivar].index[0]].to_dict()
        pred, probs = wine_model.predict(sample)
        print(f"Sample from Cultivar {cultivar + 1}: Predicted as Cultivar {pred + 1} "
              f"(Confidence: {probs[pred] * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save_model()