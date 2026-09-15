# ============================================
# AIR QUALITY PREDICTION USING RANDOM FOREST
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================
# 1. LOAD DATASET
# ============================================

file_path = "AirQualityUCI.csv"

try:
    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
        encoding="latin1"
    )

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

except Exception as e:
    print("Error while loading dataset:")
    print(e)
    exit()


# ============================================
# 2. REMOVE EMPTY COLUMNS
# ============================================

df = df.dropna(axis=1, how="all")

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nColumns:")
print(df.columns.tolist())


# ============================================
# 3. CHECK REQUIRED COLUMN
# ============================================

if "CO(GT)" not in df.columns:
    print("\nERROR: CO(GT) column not found.")
    print("Available columns:")
    print(df.columns.tolist())
    exit()


# ============================================
# 4. REPLACE MISSING VALUES
# ============================================

# In UCI dataset, -200 represents missing values
df.replace(-200, np.nan, inplace=True)

print("\nMissing values before filling:")
print(df.isnull().sum())


# ============================================
# 5. CREATE DATE AND TIME
# ============================================

if "Date" in df.columns and "Time" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Time"] = df["Time"].astype(str)

    df["DateTime"] = pd.to_datetime(
        df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"],
        errors="coerce"
    )

    # Sort by date
    df = df.sort_values("DateTime")

    # Create time features
    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Day"] = df["DateTime"].dt.day
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek


# ============================================
# 6. FILL MISSING NUMERIC VALUES
# ============================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

print("\nMissing values after filling:")
print(df.isnull().sum())


# ============================================
# 7. EXPLORATORY DATA ANALYSIS
# ============================================

# CO distribution
plt.figure(figsize=(8, 5))
plt.hist(df["CO(GT)"], bins=30)
plt.xlabel("CO Concentration")
plt.ylabel("Frequency")
plt.title("Distribution of CO Concentration")
plt.show()


# ============================================
# 8. CORRELATION HEATMAP
# ============================================

plt.figure(figsize=(12, 8))

correlation = df.select_dtypes(
    include=np.number
).corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================
# 9. CO CONCENTRATION OVER TIME
# ============================================

if "DateTime" in df.columns:

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["DateTime"],
        df["CO(GT)"]
    )

    plt.xlabel("Date")
    plt.ylabel("CO Concentration")
    plt.title("CO Concentration Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ============================================
# 10. SELECT TARGET VARIABLE
# ============================================

target = "CO(GT)"

y = df[target]


# ============================================
# 11. SELECT INPUT FEATURES
# ============================================

features = [
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH"
]

# Add time features if available
time_features = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek"
]

for column in time_features:
    if column in df.columns:
        features.append(column)


# Keep only columns that actually exist
features = [
    column for column in features
    if column in df.columns
]

print("\nInput features:")
print(features)


# ============================================
# 12. CREATE X AND Y
# ============================================

X = df[features]
y = df[target]


# ============================================
# 13. REMOVE MISSING ROWS
# ============================================

data = pd.concat([X, y], axis=1)

data = data.dropna()

X = data[features]
y = data[target]

print("\nData after removing missing values:")
print(X.shape)


# ============================================
# 14. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================
# 15. CREATE RANDOM FOREST MODEL
# ============================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ============================================
# 16. TRAIN MODEL
# ============================================

model.fit(X_train, y_train)

print("\nModel training completed!")


# ============================================
# 17. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)


# ============================================
# 18. PERFORMANCE EVALUATION
# ============================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)
print("R²   :", r2)


# ============================================
# 19. ACTUAL VS PREDICTED
# ============================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual CO")
plt.ylabel("Predicted CO")
plt.title("Actual vs Predicted CO")

plt.tight_layout()
plt.show()


# ============================================
# 20. FEATURE IMPORTANCE
# ============================================

importance = pd.Series(
    model.feature_importances_,
    index=features
)

importance = importance.sort_values(
    ascending=False
)

plt.figure(figsize=(10, 6))

importance.plot(
    kind="bar"
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================
# END
# ============================================

print("\nAir Quality Prediction completed successfully!")