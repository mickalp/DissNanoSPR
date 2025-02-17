import argparse
import joblib
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Inference script loading data from an Excel file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="DissNanoSPR_model.joblib", 
        help="Path to the joblib model file."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the Excel (.xlsx) file containing data for prediction."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Path to save the Excel file with predictions."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load the trained model
    model = joblib.load(args.model)

    # Read data from the Excel file (with extra columns if you want)
    df = pd.read_excel(args.input)


    # Predict on those features
    predictions = model.predict(df)

    # Attach predictions to the original dataframe
    df["prediction"] = predictions

    # Save or print results
    if args.output:
        df.to_excel(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        print(df[["prediction"]])

if __name__ == "__main__":
    main()
