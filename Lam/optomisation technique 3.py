# Using z-score standardisation to optimisation the model

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("merged_dataset_clean.csv")

# Composite Well-Being Score (sum of well-being indicators)
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 
                     'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
df['composite_wellbeing_score'] = df[wellbeing_columns].sum(axis=1)

# Original multiple linear regression
X = df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']]  # Screen time variables
y = df['composite_wellbeing_score']  # Well-being score

# Add a constant for intercept in statsmodels
X = sm.add_constant(X)

# Convert X to DataFrame to allow for column manipulation
X_df = pd.DataFrame(X)

# Build the linear regression using statsmodels
model = sm.OLS(y, X_df).fit()
model_report = model.summary()
print("Original Model Summary:")
print(model_report)

# Apply z-score standardization (excluding the constant column)
scaler = StandardScaler()

# Standardize the screen time variables (without the constant)
X_std = scaler.fit_transform(X_df.drop(columns=['const']))

# Convert back to DataFrame
X_std_df = pd.DataFrame(X_std, index=X_df.index, columns=X_df.columns[1:])  # Exclude 'const' in the columns

# Add the constant back after standardization
X_std_df = sm.add_constant(X_std_df)

# Rebuild the linear regression using statsmodels with standardized data
model_std = sm.OLS(y, X_std_df).fit()
model_std_report = model_std.summary()
print("Standardized Model Summary:")
print(model_std_report)
