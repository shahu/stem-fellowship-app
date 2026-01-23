import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_all_data = pd.read_csv("/Users/shahu/PycharmProjects/stem-fellowship-app/all_samples_clean_final.csv")
print(df_all_data.head())

df_all_data['REF_DATE_DT'] = pd.to_datetime(df_all_data['REF_DATE_DT'].astype(str), format='%Y%m')
df_all_data = df_all_data.set_index('REF_DATE_DT')

y = df_all_data['RNFB_w/out']
X_linear = df_all_data[['CPI_lag_1m']]

rf_exclude_cols = ['RNFB_w/out', 'RNFB_intp', 'CPI no adjusted','CPI_change_rate',
                   'diesel_price','jet_price','LE Price', 'GF Price','ZW Price','DC Price',
                   'apparent_temperature', 'temperature_2m', 'WRSI', 'FDD', 'snowfall'
                  ]
X_rf_candidate = df_all_data.drop(columns=rf_exclude_cols) # Remove all current month data, because in real-world applications for predicting current month RNBF, we cannot obtain current month data features. Testing showed no performance degradation.

'''
print("df_all_data head after processing:\n", df_all_data.head())
print("\nTarget variable 'y' head:\n", y.head())
print("\nLinear regression feature 'X_linear' head:\n", X_linear.head())
print("\nRandom forest candidate features 'X_rf_candidate' head:\n", X_rf_candidate.head())
'''


# Linear models (Linear Regression, Logistic Regression): Prioritize using Pearson to filter linearly correlated features;
# Tree models (Random Forest, XGBoost): Spearman is more suitable (tree models are sensitive to non-linear relationships and do not require distribution assumptions)

# Calculate Pearson correlations
correlations_pearson = X_rf_candidate.corrwith(y)
absolute_correlations_pearson = correlations_pearson.abs().sort_values(ascending=False)
top_20_features_pearson = absolute_correlations_pearson.head(20)

# Calculate Spearman correlations
# Note: corrwith method applies the correlation method pairwise between columns of X_rf_candidate and y
correlations_spearman = X_rf_candidate.corrwith(y, method='spearman')
absolute_correlations_spearman = correlations_spearman.abs().sort_values(ascending=False)
top_20_features_spearman = absolute_correlations_spearman.head(20)

# Create a figure with two subplots arranged in a single column
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot the top 20 Pearson correlations
sns.barplot(x=top_20_features_pearson.values, y=top_20_features_pearson.index, palette='viridis', ax=axes[0])
axes[0].set_title('Top 20 Most Correlated Features with RNFB (Pearson Correlation)')
axes[0].set_xlabel('Absolute Pearson Correlation Coefficient')
axes[0].set_ylabel('Feature')
axes[0].grid(axis='x', linestyle='--', alpha=0.7)

# Plot the top 20 Spearman correlations
sns.barplot(x=top_20_features_spearman.values, y=top_20_features_spearman.index, palette='plasma', ax=axes[1])
axes[1].set_title('Top 20 Most Correlated Features with RNFB (Spearman Correlation)')
axes[1].set_xlabel('Absolute Spearman Correlation Coefficient')
axes[1].set_ylabel('Feature')
axes[1].grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
#plt.show()
fig.savefig('/Users/shahu/PycharmProjects/stem-fellowship-app/correlation_analysis.png', dpi=300, bbox_inches='tight')

print("Top 20 Most Correlated Features with RNFB (Pearson Correlation):")
print(top_20_features_pearson)
print("\nTop 20 Most Correlated Features with RNFB (Spearman Correlation):")
print(top_20_features_spearman)


#########
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

# Define the rolling window size
window_size = 12 # months

# Initialize lists to store actual and predicted values for plotting
actual_rnbf_values_for_plot = []
hybrid_predicted_values_for_plot = []

# Initialize lists to store metrics for training and testing
train_rmse_scores = []
train_mae_scores = []
train_r2_scores = []
test_rmse_scores = []
test_mae_scores = []
test_r2_scores = []

# Iterate through the dataset using a rolling window
# The loop starts after the initial window size, and goes up to the second to last element to ensure a test set of at least 1 month
for i in range(window_size, len(df_all_data) - 3): # changed to len(df_all_data) - 3 to accommodate 3 month prediction
    # Split data into training and test sets
    train_data = df_all_data.iloc[:i]
    test_data = df_all_data.iloc[i:i+3] # Predict 3 month ahead

    y_train = y.iloc[:i]
    X_linear_train = X_linear.iloc[:i]
    X_rf_candidate_train = X_rf_candidate.iloc[:i]

    y_test = y.iloc[i:i+3] # next quarter is 3 months
    X_linear_test = X_linear.iloc[i:i+3] # next quarter is 3 months
    X_rf_candidate_test = X_rf_candidate.iloc[i:i+3] # next quarter is 3 months

    # 1. Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_linear_train, y_train)
    lr_pred_test = lr_model.predict(X_linear_test)
    lr_pred_train = lr_model.predict(X_linear_train)

    # 2. Calculate residuals from Linear Regression on the training set
    residuals_train = y_train - lr_pred_train

    # 3. Perform permutation importance on candidate random forest features using residuals as target
    # Drop rows with NaN values if any, as permutation importance doesn't handle them
    X_rf_candidate_train_cleaned = X_rf_candidate_train.dropna(axis=1)
    residuals_train_aligned = residuals_train[X_rf_candidate_train_cleaned.index]

    rf_pred_test = 0 # Default to 0 if RF cannot be trained
    rf_pred_train = np.zeros_like(residuals_train_aligned) # Default to zeros for train residuals

    # Ensure at least one feature remains after dropping NaNs
    if X_rf_candidate_train_cleaned.empty or len(X_rf_candidate_train_cleaned.columns) == 0:
        # If no valid features, skip RF part or use a simpler fallback
        pass
    else:
        # Create a dummy RF for permutation importance, can be lightweight
        dummy_rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
        dummy_rf.fit(X_rf_candidate_train_cleaned, residuals_train_aligned)

        # Permutation importance
        result = permutation_importance(dummy_rf, X_rf_candidate_train_cleaned, residuals_train_aligned, n_repeats=5, random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()[::-1]
        top_10_features = X_rf_candidate_train_cleaned.columns[sorted_idx[:10]].tolist()  # try differnt count of the features 10 or 20 or 30

        # Ensure top_10_features are present in the test set
        top_10_features_test = [f for f in top_10_features if f in X_rf_candidate_test.columns]
        top_10_features_train = [f for f in top_10_features if f in X_rf_candidate_train_cleaned.columns]

        if not top_10_features_test or not top_10_features_train:
            rf_pred_test = 0
            rf_pred_train = np.zeros_like(residuals_train_aligned)
        else:
            # 4. Train Random Forest Regressor on top 10 features to predict residuals
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_rf_candidate_train_cleaned[top_10_features_train], residuals_train_aligned)
            rf_pred_test = rf_model.predict(X_rf_candidate_test[top_10_features_test])
            rf_pred_train = rf_model.predict(X_rf_candidate_train_cleaned[top_10_features_train])

    # 5. Combine predictions
    hybrid_pred_test = lr_pred_test + rf_pred_test
    hybrid_pred_train = lr_pred_train + rf_pred_train

    # Store results for plotting (only the first prediction for each window)
    actual_rnbf_values_for_plot.append(y_test.values[0])
    hybrid_predicted_values_for_plot.append(hybrid_pred_test[0])

    # Calculate and store training metrics
    train_rmse_scores.append(np.sqrt(mean_squared_error(y_train, hybrid_pred_train)))
    train_mae_scores.append(mean_absolute_error(y_train, hybrid_pred_train))
    train_r2_scores.append(r2_score(y_train, hybrid_pred_train))

    # Calculate and store test metrics
    test_rmse_scores.append(np.sqrt(mean_squared_error(y_test, hybrid_pred_test)))
    test_mae_scores.append(mean_absolute_error(y_test, hybrid_pred_test))
    test_r2_scores.append(r2_score(y_test, hybrid_pred_test))

# Save the models from the last window
joblib.dump(lr_model, '/Users/shahu/PycharmProjects/stem-fellowship-app/lr_model.pkl')
joblib.dump(rf_model, '/Users/shahu/PycharmProjects/stem-fellowship-app/rf_model.pkl')

print("\n--- Final Models Saved ---")
print(f"Linear Regression model saved to: /Users/shahu/PycharmProjects/stem-fellowship-app/lr_model.pkl")
print(f"Random Forest model saved to: /Users/shahu/PycharmProjects/stem-fellowship-app/rf_model.pkl")

print("\n--- Linear Regression Parameters ---")
print(f"Coefficients: {lr_model.coef_}")
print(f"Intercept: {lr_model.intercept_}")
print(f"Hyperparameters: {lr_model.get_params()}")

print("\n--- Random Forest Parameters ---")
print(f"Hyperparameters: {rf_model.get_params()}")
# Optional: also show top features for the last RF model
print(f"Features used in last RF: {top_10_features_train}")

print("First 5 actual RNFB values:", actual_rnbf_values_for_plot[:5])
print("First 5 hybrid predicted RNFB values:", hybrid_predicted_values_for_plot[:5])
print("Number of actual values:", len(actual_rnbf_values_for_plot))
print("Number of predicted values:", len(hybrid_predicted_values_for_plot))

# Summarize average metrics across all rolling windows
print("\n--- Average Metrics Across Rolling Windows ---")
print(f"Average Training RMSE: {np.mean(train_rmse_scores):.2f}")
print(f"Average Training MAE: {np.mean(train_mae_scores):.2f}")
print(f"Average Training R-squared: {np.mean(train_r2_scores):.2f}")



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Get the dates corresponding to the predictions
# The predictions start from the 'window_size' index up to 'len(df_all_data) - 1'
print(window_size)
print(len(df_all_data))

prediction_dates = df_all_data.index[window_size : len(df_all_data) - 3]  # next quarter is 3 months

# Create a DataFrame for plotting
results_df = pd.DataFrame({
    'Date': prediction_dates,
    'Actual RNFB': actual_rnbf_values_for_plot,
    'Hybrid Predicted RNFB': hybrid_predicted_values_for_plot
})

# Set 'Date' as index for plotting
results_df.set_index('Date', inplace=True)
results_df.to_csv('/Users/shahu/PycharmProjects/stem-fellowship-app/actual_vs_hybrid_predicted_rnfb.csv')

# Plot the results
plt.figure(figsize=(15, 7))
sns.lineplot(data=results_df[['Actual RNFB', 'Hybrid Predicted RNFB']])
plt.title('Actual vs. Hybrid Model Predicted RNFB Over Time')
plt.xlabel('Date')
plt.ylabel('RNFB_w/out Value')
plt.legend(title='Prediction Type')
plt.grid(True)
plt.tight_layout()
#plt.show()

# save the plot
plt.savefig('/Users/shahu/PycharmProjects/stem-fellowship-app/actual_vs_hybrid_predicted_rnfb.png', dpi=300, bbox_inches='tight')

# Conclude by summarizing the performance
# Calculate evaluation metrics if needed, for a more quantitative summary
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(actual_rnbf_values_for_plot, hybrid_predicted_values_for_plot)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_rnbf_values_for_plot, hybrid_predicted_values_for_plot)
r2 = r2_score(actual_rnbf_values_for_plot, hybrid_predicted_values_for_plot)

# Summarize average metrics across all rolling windows
print("\n--- Average Metrics Across Rolling Windows ---")
print(f"Average Training RMSE: {np.mean(train_rmse_scores):.2f}")
print(f"Average Training MAE: {np.mean(train_mae_scores):.2f}")
print(f"Average Training R-squared: {np.mean(train_r2_scores):.2f}")

print(f"\n--- Hybrid Model Performance Summary ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print(f"\nInsights: The plot visually demonstrates how closely the hybrid model's predictions track the actual values. \nThe model appears to capture the overall trend, but there might be deviations during periods of high volatility or sudden changes. The calculated metrics provide a quantitative measure of accuracy, with R2 indicating the proportion of variance in the actual values predictable from the model.")
