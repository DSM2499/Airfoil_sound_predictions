# ML Pipeline for Airfoil Noise Prediction
A SparkML implementation, using various aspects of Apache Spark and SparkML

## Summary
The Jupyter notebook offers a comprehensive guide to building and deploying a machine learning pipeline for predicting airfoil noise. It demonstrates essential data engineering tasks, streamlined pipeline creation for model training, thorough evaluation to assess model effectiveness, and techniques for model persistence. This framework not only supports the project's immediate goals but also provides a scalable template for similar predictive modeling tasks in the aeronautical industry.

## ETL Activities:

### Loading Data
- **Description**: The notebook starts with loading the airfoil dataset from a CSV file into a DataFrame. This step is crucial for accessing the data in a structured format suitable for manipulation and analysis.

### Data Cleaning
- **Duplicate Removal**
  - **Method**: `dropDuplicates()`
  - **Description**: Ensures the uniqueness of each data entry, preventing redundancy which could skew the model training.
- **Null Value Handling**
  - **Method**: `dropna()`
  - **Description**: Removes rows containing missing values, ensuring the dataset's integrity.
- **Data Transformation**
  - **Description**: Specific transformations are not detailed in the initial extraction, but typical methods might include normalizing or scaling features, which are essential for many ML models.
- **Data Storing**
  - **Method**: `DataFrame.write.parquet()`
  - **Description**: Post-cleaning, the data is stored in Parquet format, which offers efficient storage and fast retrieval.

## Machine Learning Pipeline Creation

### Pipeline Architecture
- **Description**: The ML pipeline is constructed using Spark's Pipeline class, which allows for sequential processing of data from preprocessing to model training.

### Transformers and Estimators
- **Transformers**
  - **Description**: Used for feature engineering, like vector assembling of features into a single column using VectorAssembler. This is crucial for preparing the dataset for ML models in Spark.
- **Estimators**
  - **Model**: LinearRegression
  - **Description**: The primary estimator here is a linear regression model, which predicts the sound level based on other features.

### Pipeline Execution
- **Description**: The configured pipeline encapsulates both data transformation and model training steps, which are executed in sequence when the pipeline is run.

## Model Evaluation

### Evaluation Metrics
- **Mean Squared Error (MSE)**
  - **Description**: Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
- **Mean Absolute Error (MAE)**
  - **Description**: A measure of errors between paired observations expressing the same phenomenon.
- **R-Squared (R2)**
  - **Description**: Provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model.

### Evaluation Output
- **Description**: These metrics are computed and printed to give insights into the model's performance, guiding potential improvements.

## Persist the Model

### Model Saving
- **Method**: `pipelineModel.write().save("Final_Project")`
- **Description**: The entire pipeline model is saved to a directory. This method ensures that the model configuration and learned parameters are stored for future use.

### Model Loading
- **Method**: `PipelineModel.load("Final_Project")`
- **Description**: The saved model is reloaded from the directory, which enables the model to be used for further predictions without retraining.

### Making Predictions
- **Description**: The loaded model is then used to make predictions on new or test data to assess its practical applicability.
