import joblib
import numpy as np
import argparse

def load_model(model_path: str):
    """
    Loads a trained model from a .joblib file.
    """
    model = joblib.load(model_path)
    return model

def parse_arguments():
    """
    Parses command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Inference script for an SVC model')
    parser.add_argument('--model', type=str, default='my_svc_model.joblib',
                        help='Path to the joblib model file')
    parser.add_argument('--input', type=str, required=True,
                        help='Comma-separated list of numeric input features (e.g., "5.1,3.5,1.4,0.2")')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load the model
    model = load_model(args.model)

    # Convert the comma-separated string into a NumPy array
    # For example, if your input is "5.1,3.5,1.4,0.2", this becomes [5.1, 3.5, 1.4, 0.2]
    features = np.array([float(x) for x in args.input.split(',')]).reshape(1, -1)

    # Perform prediction
    prediction = model.predict(features)

    # Print the prediction
    # Depending on your model, this might be a class label (e.g., "Iris-setosa") or numeric value
    print("Prediction:", prediction[0])

if __name__ == '__main__':
    main()

