import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("student_data.csv")

print("Initial Dataset Shape:", df.shape)
print("\n")

# 1. Display first 5 rows
print("1. Head of Dataset")
print(df.head())
print("\n")

# 2. Dataset info
print("2. Dataset Info")
print(df.info())
print("\n")

# 3. Rename columns (make lowercase)
df.columns = [col.lower() for col in df.columns]
print("3. Renamed Columns")
print(df.columns)
print("\n")

# 4. Check missing values
print("4. Missing Values Count")
print(df.isnull().sum())
print("\n")

# 5. Fill missing values with mean (numeric only)
df.fillna(df.mean(numeric_only=True), inplace=True)
print("5. Missing Values Filled (Mean)")
print(df.tail())
print("\n")

# 6. Remove duplicate rows
df.drop_duplicates(inplace=True)
print("6. Duplicates Removed - Shape")
print(df.shape)
print("\n")

# 7. Detect outliers using IQR (example on first numeric column)
num_col = df.select_dtypes(include=np.number).columns[0]
Q1 = df[num_col].quantile(0.25)
Q3 = df[num_col].quantile(0.75)
IQR = Q3 - Q1
df = df[(df[num_col] >= Q1 - 1.5 * IQR) & (df[num_col] <= Q3 + 1.5 * IQR)]
print(f"7. Outliers Removed from column '{num_col}'")
print(df.tail())
print("\n")

# 8. Data type conversion (if any object numeric)
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
print("8. Data Type Conversion Attempted")
print(df.dtypes)
print("\n")

# 9. Feature scaling (Min-Max)
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = (df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min())
print("9. Min-Max Scaling Applied")
print(df.tail())
print("\n")

# 10. Normalization (L2 norm)
df[num_cols] = df[num_cols].div(np.sqrt((df[num_cols] ** 2).sum(axis=1)), axis=0)
print("10. Normalization Applied")
print(df.tail())
print("\n")

# 11. Binning (first numeric column)
df[num_col + "_bin"] = pd.cut(df[num_col], bins=3, labels=["Low", "Medium", "High"])
print("11. Binning Applied")
print(df[[num_col, num_col + "_bin"]].tail())
print("\n")

# 12. Aggregation (mean of numeric columns)
print("12. Aggregation - Mean")
print(df[num_cols].mean())
print("\n")

# 13. Mean calculation
print("13. Mean")
print(df[num_cols].mean())
print("\n")

# 14. Median calculation
print("14. Median")
print(df[num_cols].median())
print("\n")

# 15. Mode calculation
print("15. Mode")
print(df.mode().head())
print("\n")

# 16. One-Hot Encoding (first categorical column if exists)
cat_cols = df.select_dtypes(include='object').columns
if len(cat_cols) > 0:
    df = pd.get_dummies(df, columns=[cat_cols[0]])
    print("16. One-Hot Encoding Applied")
    print(df.tail())
else:
    print("16. No Categorical Column for One-Hot Encoding")
print("\n")
df.fillna(df.mean(numeric_only=True), inplace=True)

print("17. Histogram of Age")
plt.figure()
plt.hist(df['age'])
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Histogram of Age")
plt.show()

print("19. Scatter Plot: Study Time vs Final Grade")

plt.figure()
plt.scatter(df['studytime'], df[grade_col])
plt.xlabel("Study Time")
plt.ylabel("Final Grade")
plt.title("Study Time vs Final Grade")
plt.show()

print("20. Bar Chart: Average Final Grade by Gender")

avg_grade_gender = df.groupby('sex')[grade_col].mean()
plt.figure()
plt.bar(avg_grade_gender.index, avg_grade_gender.values)
plt.xlabel("Gender")
plt.ylabel("Average Final Grade")
plt.title("Average Final Grade by Gender")
plt.show()
