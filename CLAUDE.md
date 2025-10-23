# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Airline Customer Satisfaction Analysis** project that performs comprehensive data analysis on airline customer data. The project uses machine learning to predict customer satisfaction, analyze sentiment from reviews, and understand booking patterns.

**Primary Technologies:**
- Python 3.13
- Jupyter Notebooks (.ipynb)
- Machine Learning: scikit-learn, TensorFlow/Keras
- Data Analysis: pandas, numpy
- Visualization: matplotlib, seaborn
- NLP: VADER Sentiment Analysis, SHAP, LIME

## Development Environment

**Setup:**
```bash
# The project uses a virtual environment located at .venv/
source .venv/bin/activate  # On macOS/Linux

# Install dependencies (if needed)
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow vaderSentiment shap lime jupyter
```

**Running Jupyter Notebooks:**
```bash
# Start Jupyter notebook server
jupyter notebook

# Or run a specific notebook
jupyter nbconvert --to notebook --execute ACL_project_F.ipynb
```

## Dataset Structure

The project analyzes **4 datasets** located in the `dataset/` directory:

1. **AirlineScrappedReview_Cleaned.csv** (3,575 records)
   - Customer reviews with ratings (1-10 scale)
   - Contains: Route, Rating, Traveller_Type, Class, Review_content, Verified status
   - Used for: Sentiment analysis and satisfaction prediction

2. **Passanger_booking_data.csv** (50,002 records)
   - Booking patterns and completion status
   - Contains: flight_hour, route, booking_origin, purchase_lead, flight_duration
   - Used for: Booking completion prediction and flight hour distribution analysis

3. **Customer_comment.csv** (9,424 records)
   - Food & beverage feedback (pre-computed sentiment, but flawed - mostly "Neutral")
   - Note: This dataset was **dropped** in final analysis due to poor quality

4. **Survey data_Inflight Satisfaction Score.csv** (47,074 records)
   - Inflight satisfaction survey scores
   - Note: This dataset was **dropped** in final analysis as not required for objectives

**Active Datasets:** Only datasets #1 and #2 are used in the final analysis pipeline.

## Project Architecture

### Primary Notebook: `ACL_project_F.ipynb`

This is the **main deliverable** containing the complete analysis pipeline:

**Pipeline Structure:**
1. **Data Loading & Cleaning**
   - Duplicate removal
   - Missing value handling
   - Data type validation

2. **Data Engineering**
   - Sentiment Analysis (VADER on Review_content)
   - Top 10 flight routes analysis (cross-dataset)
   - Flight hour booking distribution
   - Review patterns by Traveller_Type × Class

3. **Feature Engineering**
   - Target variable: `satisfaction` (1 if Rating ≥ 5, else 0)
   - High-cardinality handling: Top 15 categories for Route/Start_Location/End_Location → "Other"
   - One-hot encoding for categorical features
   - StandardScaler normalization

4. **Predictive Modeling**
   - **Model 1:** Logistic Regression (baseline)
   - **Model 2:** Shallow FFNN (2 hidden layers: 32→16 neurons)
   - **Model 3:** Deep FFNN (4 hidden layers: 128→64→32→16 neurons, with BatchNorm + Dropout)
   - All models use early stopping with validation split

5. **Model Explainability**
   - SHAP (TreeExplainer for feature importance)
   - LIME (Local interpretability for individual predictions)

6. **Inference Function**
   - `predict_satisfaction()`: Takes sample data, returns predictions from all 3 models
   - `display_prediction_results()`: Visualizes prediction comparison

### Secondary Notebook: `Comprehensive_Airline_Analysis.ipynb`

Earlier exploratory analysis covering all 4 datasets (less refined than ACL_project_F).

## Key Implementation Details

### Sentiment Analysis
- **Tool:** VADER (vaderSentiment.SentimentIntensityAnalyzer)
- **Method:** Compound score on `Review_content` column
- **Thresholds:**
  - Positive: score ≥ 0.5
  - Negative: score ≤ -0.5
  - Neutral: -0.5 < score < 0.5
- **Output:** New columns `sentiment_score` and `sentiment_category`

### Feature Selection Strategy
**Selected Features:**
- `Traveller_Type` (Business/Solo/Family/Couple Leisure)
- `Class` (Economy/Business/Premium Economy/First Class)
- `Route` (e.g., "London to Paris")
- `Start_Location`, `End_Location`
- `Verified` (Trip Verified / Not Verified)
- `sentiment_score` (computed via VADER)

**Excluded Features:**
- `Passanger_Name`, `Flying_Date`, `Review_title`, `Review_content` (raw text)
- `Layover_Route` (too many missing values)
- Latitude/Longitude/Address fields (redundant with Route)
- `Rating` (target leakage), `sentiment_category` (redundant with score)

### High-Cardinality Feature Handling
```python
# Keep top 15 most common values, group rest as "Other"
N_TOP_CATEGORIES = 15
high_cardinality_features = ['Route', 'Start_Location', 'End_Location']

for feature in high_cardinality_features:
    top_categories = df[feature].value_counts().nlargest(N_TOP_CATEGORIES).index
    df[feature] = df[feature].where(df[feature].isin(top_categories), 'Other')
```

### Model Training Best Practices
- **Data Split:** 80% train, 20% test (stratified by target)
- **Validation Split (for NNs):** Further split train into 80% train, 20% validation
- **Early Stopping:** patience=15 epochs, monitor='val_loss', restore_best_weights=True
- **Neural Network Architecture:**
  - Activation: ReLU (hidden), Sigmoid (output)
  - Regularization: Dropout (0.1-0.3) + BatchNormalization
  - Optimizer: Adam (lr=0.001)
  - Loss: binary_crossentropy

### Model Performance (Current Results)
- **Model 1 (Logistic):** ~78-80% accuracy
- **Model 2 (Shallow NN):** ~79-82% accuracy
- **Model 3 (Deep NN):** ~80-83% accuracy
- **Key Insight:** sentiment_score is by far the most important feature (~69% importance)

## Common Development Tasks

### Running the Full Analysis Pipeline
```bash
# Activate virtual environment
source .venv/bin/activate

# Launch Jupyter and open ACL_project_F.ipynb
jupyter notebook ACL_project_F.ipynb

# Run all cells in order (Kernel → Restart & Run All)
```

### Making Predictions on New Data
```python
# After running ACL_project_F.ipynb to train models:

# Prepare sample data (must include all required features)
sample = pd.DataFrame({
    'Review_content': ['Amazing flight, great service!'],
    'Rating': [9],
    'Traveller_Type': ['Business'],
    'Class': ['Business Class'],
    'Route': ['London to New York'],
    'Start_Location': ['London'],
    'End_Location': ['New York'],
    'Verified': ['Trip Verified']
})

# Run inference
predictions = predict_satisfaction(
    sample_data=sample,
    log_model=log_model,
    ffnn_model=ffnn_model,
    model3=model3,
    scaler=scaler,
    X_features=X_features,
    top_categories_dict=top_categories_dict
)

# Display results
display_prediction_results(sample, predictions)
```

### Modifying the Model Architecture

When changing Model 3 (Deep NN) architecture:

1. Locate the model definition cell (search for "MODEL 3: FFNN WITH EARLY STOPPING")
2. Modify the `Sequential` layers:
   ```python
   model3 = Sequential([
       Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
       BatchNormalization(),
       Dropout(0.3),
       # Add/remove layers here
       Dense(1, activation='sigmoid')
   ])
   ```
3. Adjust early stopping patience if needed
4. Re-run training cells

### Adding New Features

To add a new feature to the model:

1. Add to `relevant_features` dict in feature engineering section
2. Update `modeling_features` list
3. Ensure proper handling in preprocessing (missing values, encoding)
4. If high-cardinality, add to `high_cardinality_features` list
5. Re-run feature engineering and model training cells

## Data Quality Notes

**Known Issues:**
- **Customer_comment.csv:** Pre-computed sentiments are unreliable (0% positive, should be ~30-40%)
- **Survey data:** String-to-numeric conversion issues with 'score' column
- **Missing values:** ~73% of reviews missing `Flying_Date`, ~20% missing `Route`

**Dataset Selection Rationale:**
- AirlineScrappedReview_Cleaned.csv: High quality, comprehensive features ✓
- Passanger_booking_data.csv: Clean data, minimal missing values ✓
- Customer_comment.csv: Dropped (poor sentiment labeling, redundant scope) ✗
- Survey data: Dropped (not required for objectives, data type issues) ✗

## Reporting

The project includes a comprehensive markdown report: `Airline_Data_Analysis_Report.md`

This report documents:
- Complete methodology and rationale
- Data cleaning decisions
- Analysis findings for all requirements
- Model performance metrics
- Business recommendations
- Implementation steps

## Architecture Insights

**Key Design Decisions:**

1. **Why VADER over other sentiment tools?**
   - Designed for social media/review text
   - Handles context, punctuation, capitalization
   - No training required (rule-based)
   - Fast and lightweight

2. **Why Random Forest for feature importance?**
   - Provides interpretable importance scores
   - Handles mixed data types well
   - Robust to outliers
   - Used as preliminary analysis before deep learning

3. **Why three models?**
   - Model 1 (Logistic): Baseline, interpretable, fast inference
   - Model 2 (Shallow NN): Balance of complexity and performance
   - Model 3 (Deep NN): Maximum performance, explores capacity limits
   - Ensemble insights from comparing all three

4. **Why StandardScaler over MinMaxScaler?**
   - Features have different scales (sentiment: -1 to 1, ratings: 1-10)
   - Neural networks prefer zero-centered data
   - Robust to outliers compared to MinMax

## File Organization

```
ACL Project/
├── dataset/
│   ├── AirlineScrappedReview_Cleaned.csv    # Primary dataset
│   ├── Passanger_booking_data.csv           # Booking data
│   ├── Customer_comment.csv                 # (Dropped)
│   └── Survey data_Inflight Satisfaction Score.csv  # (Dropped)
├── ACL_project_F.ipynb                      # Main analysis notebook
├── Comprehensive_Airline_Analysis.ipynb     # Exploratory analysis
├── Airline_Data_Analysis_Report.md          # Final report
└── .venv/                                   # Virtual environment
```

## Important Context for AI Assistants

When modifying this project:

1. **Always maintain the data pipeline order:** Load → Clean → Engineer → Model → Explain → Infer
2. **Feature engineering is critical:** The high-cardinality handling prevents dimensionality explosion
3. **VADER must be applied before modeling:** sentiment_score is the most important feature
4. **Keep stratified splits:** The target is imbalanced (~50-50 satisfied/dissatisfied)
5. **Early stopping is essential:** Prevents overfitting in neural networks
6. **Use the inference function:** Don't bypass preprocessing - it must match training exactly

**Common Pitfalls:**
- Forgetting to apply VADER to new samples before prediction
- Not grouping high-cardinality features into "Other" for unseen categories
- Skipping StandardScaler transformation on inference data
- Not aligning one-hot encoded columns between training and inference
- Changing preprocessing steps without retraining models
