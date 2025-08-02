# World Happiness Score Prediction Using Machine Learning

This is a machine learning project made through the AI fellowship with Cornell Tech. It predicts the World Happiness Report's "Life Ladder" happiness score for various countries based on socioeconomic, health, and social indicators from the 2018 dataset. 

The project explores multiple regression methods including Linear Regression, Ridge and Lasso regularization, and Random Forest Regression. After tuning and optimization, the Random Forest model achieved the best accuracy with an R² score of approximately 0.88, outperforming the linear models which hovered around 0.75.

---

## Dataset

- Source: WHR2018Chapter2OnlineData.csv (World Happiness Report 2018)
- Contains features such as GDP per capita, social support, healthy life expectancy, freedom to make life choices, generosity, perceptions of corruption, positive/negative affect, confidence in national government, etc.
- Target variable (label): **Life Ladder** — a continuous score measuring national happiness.

---

## Problem Definition

- **Type:** Supervised regression problem.
- **Goal:** Predict continuous happiness score based on multiple socioeconomic and social indicators.
- **Importance:** 
  - Enables governments and organizations to understand and improve factors influencing happiness.
  - Helps businesses tailor strategies to different cultural and socioeconomic environments.

---

## Data Preparation & Exploration

- Dropped features with many missing values (GINI index, Democratic Quality, Delivery Quality).
- Imputed remaining missing values using median imputation for robustness.
- Applied winsorization (1st and 99th percentiles) to reduce outliers while preserving data distribution.
- Scaled features using `StandardScaler` to normalize ranges.
- Explored distributions, checked skewness, and visualized outliers using histograms and boxplots.

---

## Models & Training

- Models tested:
  - Linear Regression
  - Ridge Regression (with hyperparameter tuning via GridSearchCV)
  - Lasso Regression (with hyperparameter tuning via GridSearchCV)
  - Random Forest Regressor (with hyperparameter tuning via GridSearchCV)

- Evaluation metrics:
  - R² Score (Coefficient of Determination)
  - Mean Squared Error (MSE)

- Cross-validation (5-fold) used during hyperparameter tuning to reduce overfitting and select best models.

---

## Results

- Linear Regression baseline achieved a mean CV R² ≈ 0.74.
- Ridge and Lasso Regression with tuned alpha values improved R² to about 0.78 on the test set.
- Random Forest significantly improved performance:
  - Best CV R² ≈ 0.85
  - Test R² ≈ 0.88
  - Test MSE reduced by ~40% compared to linear models.

---

## Conclusions & Next Steps

- Non-linear models like Random Forest capture complex relationships in the data better than linear methods.
- The pipeline includes robust preprocessing to handle missing data and outliers.
- Future work could explore more advanced models (e.g., Gradient Boosting, Neural Networks) or deeper feature engineering.
- Model interpretability and feature importance analysis can provide insights into key happiness drivers.

---

## Usage

1. Clone the repository.
2. Ensure dependencies are installed (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`).
3. Run the Jupyter notebook or Python scripts to reproduce data preprocessing, model training, and evaluation.
4. Modify model parameters or try new models as desired.

---

## Author

[Your Name]

---

## References

- World Happiness Report 2018 Dataset: [Link if available]
- Scikit-learn documentation: https://scikit-learn.org
