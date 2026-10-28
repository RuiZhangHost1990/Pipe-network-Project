The code is used to predict the 'interaction coefficient' for corroded pipe elbows. The main code for training-prediction using five different AI models is: PyTabKit_benchmark_10fold_predict_v4.py. And the results is stored in Predict_with_predictions_v4.csv.

Models and Evaluation

The repository includes implementations of several machine-learning and deep-learning regression algorithms for structured tabular data:

XGBoost

CatBoost

RealMLP

TabM

xRFM

The benchmarking workflow uses a training–validation split and 10-fold cross-validation for model evaluation and, where applicable, hyperparameter selection. Model performance is assessed using commonly adopted regression metrics, including:

Coefficient of determination ((R^2))

Root mean square error (RMSE)

Mean absolute error (MAE)

10-fold cross-validation RMSE

Installation

The scripts require Python and several machine-learning libraries. A Python environment can be created using Conda:

conda create -n pipe-interaction python=3.10
conda activate pipe-interaction

Install the required packages:

pip install numpy pandas scikit-learn
pip install xgboost catboost lightgbm
pip install pytabkit

Please refer to the official PyTorch and PyTabKit documentation for any additional installation requirements associated with the selected models.

Usage

1. Interaction coefficient prediction

Run the main prediction script:

python PyTabKit_benchmark_10fold_predict_v4.py

For the updated dataset, use the corresponding updated script once it has been added to the repository:

python PyTabKit_benchmark_10fold_predict_v4_newdata.py

Before running either script, update the dataset path and check that the input columns match the feature definitions in the code.

2. Model benchmarking

To compare the performance of the five regression models, run:

python SuperPyTabKit_benchmark_10fold.py

The script reports the evaluation metrics for each model and summarises their predictive performance on the validation set.

3. Prediction results

The prediction results from the v4 workflow are provided in:

Predict_with_predictions_v4.csv

Reproducibility

The scripts and configuration details are provided to support transparency and reproducibility of the computational workflow. However, exact reproduction of the reported results requires access to the original dataset and the corresponding software environment.

Users should also ensure that the dataset preprocessing, feature definitions, random seeds, and model configurations are consistent with those used in the original experiments.

Acknowledgements

This research was supported by the Innovation and Technology Fund, under the Innovation and Technology Support Programme (ITSP), and the Water Supplies Department of the Hong Kong Special Administrative Region (Grant No. ITS/067/19FP).
