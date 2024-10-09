# Using Gaussian transformation to optimisation the model

import pandas as pd
import statsmodels.api as sm

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

#apply Power Transfomer
from sklearn.preprocessing import PowerTransformer

scaler = PowerTransformer()

# remove the column name
X_df = X_df.drop(["const"], axis=1)

X_pow = scaler.fit_transform(X.values)
#print(X_std)

# bring the column name back
df_X_pow = pd.DataFrame(X_pow, index=X.index, columns=X.columns)

#print(df_X_std.info())
#print(df_X_std.head())

# rebuild the linear regression using statsmodels + PowerTransfomer
df_X_pow = sm.add_constant(df_X_pow)
model= sm.OLS(y, df_X_pow).fit()
model_report = model.summary()
print(model_report)
