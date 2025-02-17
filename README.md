# DissNanoSPR

# Dissolution Prediction Model for Nanomaterials in Natural Waters

This repository contains a machine learning model that predicts the dissolution of nanomaterials in natural waters based on various physico-chemical properties.

## Requirements

- Python 3.10
- [pandas](https://pypi.org/project/pandas/)  
- [NumPy](https://pypi.org/project/numpy/)  
- [scikit-learn](https://pypi.org/project/scikit-learn/)  
- [joblib](https://pypi.org/project/joblib/)  



## Usage

1. **Clone the repository** to your local machine:
   ```bash
   git clone https://github.com/your-username/nanomaterial-dissolution.git
   cd nanomaterial-dissolution
   ```

2. **Run the script**:
   ```bash
   python main.py --path_to_csv_file <PATH_TO_CSV>
   ```
   Replace `<PATH_TO_CSV>` with the path to your input CSV file.

## Input Data Format

The CSV file must contain **exactly 8 columns** corresponding to the following variables (in this order):

1. **Solvation enthalpy**  
2. **Bond dissociation enthalpy**  
3. **Primary size**  
4. **Core-to-valence electron ratio**  
5. **pH**  
6. **Total concentration in medium**  
7. **Temperature**  
8. **Presence of light** (where `1` indicates the *absence* of light, and `0` indicates the *presence* of light; `0` should be used if unsure)

> **Note**: If your data includes any additional columns, please remove or ignore them before running the prediction.

## Output

After running the script, the predictions for the dissolution of each nanomaterial listed in your CSV will be displayed or saved.

## Future Updates

- A reference table with recommended dissociation and solvation enthalpy values will be added soon to aid in data preparation.

---

For questions or issues, please open a GitHub Issue or contact the repository maintainer.
