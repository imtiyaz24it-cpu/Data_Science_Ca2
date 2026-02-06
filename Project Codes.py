# 1. head() command
print("\n1. head() - First 5 rows:")
print(df.head())
print()

# 2. info() command
print("2. info() - Dataset information:")
df.info()
print()

# 3. describe() command
print("3. describe() - Statistical summary:")
print(df.describe())
print()

# 4. shape command
print("4. shape - Dataset dimensions:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print()

# 1. Data Cleaning Techniques: 
Technique 1: Missing Values Check
print("Missing Values Per Column:")
print(df.isnull().sum())

#Technique 2: Duplicate Detection
dup_count = df.duplicated().sum()
print("Total Duplicate Rows:", dup_count



# Technique 3: Duplicate Removal
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)

# Technique 4: Column Standardization
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("Standardized Columns:")
print(df.columns.tolist())

#Technique 5: Data Type Correction
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
print(df[['age', 'balance']].dtypes)

#2. Data Transformation Techniques: 

#Technique 6: Date Feature Extraction

df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_name'] = df['date'].dt.day_name()
print(df[['date', 'year', 'month', 'day', 'day_name']].head())

# Technique 7: Time Feature Extraction
# Dummy time column
df['time'] = pd.to_datetime(
    np.random.randint(0, 24, size=len(df)),
    unit='h'
).strftime('%H:%M:%S') # Removed .dt here
df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
print(df[['time', 'hour']].head())

# Technique 8: Time Categorization
def time_category(hour):
    if hour < 12:
        return 'Morning'
    elif hour < 17:
        return 'Afternoon'
    else:
        return 'Night'
df['time_category'] = df['hour'].apply(time_category)
print(df['time_category'].value_counts())

# Technique 9: Binary Encoding
df['y_binary'] = df['deposit'].apply(lambda x: 1 if x == 'yes' else 0)
print(df[['deposit', 'y_binary']].head())


#3. Feature Engineering Techniques: 

# Technique 10: Price Categorization (Balance)
def balance_category(balance):
    if balance < 0:
        return 'Low'
    elif balance < 5000:
        return 'Medium'
    else:
        return 'High'

df['balance_category'] = df['balance'].apply(balance_category)
print(df['balance_category'].value_counts())


# Technique 11: Coffee Grouping (Text Grouping using job)
def job_group(job):
    job = job.lower()
    if 'admin' in job:
        return 'Admin'
    elif 'tech' in job:
        return 'Technical'
    else:
        return 'Other'

df['job_group'] = df['job'].apply(job_group)
print(df[['job', 'job_group']].head())

# Technique 12: Peak Hour Indicator
peak_hours = [9, 10, 11, 16, 17]
df['peak_hour'] = df['hour'].apply(lambda x: 1 if x in peak_hours else 0)

print(df['peak_hour'].value_counts())

# Technique 13: Daily Sales Totals
daily_sales = df.groupby('date')['balance'].sum().reset_index()
df = df.merge(daily_sales, on='date', suffixes=('', '_daily_total'))

print(df[['date', 'balance_daily_total']].head())

# 4. Exploratory Data Analysis: 

# Technique 14: Sales Distribution Histogram
plt.figure(figsize=(7,4))
plt.hist(df['balance'], bins=30, edgecolor='black')
plt.title('Balance Distribution')
plt.show()

# Technique 15: Popular Category Bar Chart
df['job'].value_counts().head(10).plot(kind='bar', figsize=(7,4))
plt.title('Top 10 Jobs')
plt.show()

# Technique 16: Daily Trend Line Chart

df.groupby('date')['balance'].sum().head(30).plot(kind='line', figsize=(9,4))
plt.title('Daily Balance Trend')
plt.show()



# Technique 17: Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 5. Text Data Techniques: 
Technique 18: Text Length Analysis
df['job_length'] = df['job'].str.len()
print(df['job_length'].describe())

plt.hist(df['job_length'], bins=20, edgecolor='black')
plt.title('Job Text Length Distribution')
plt.show()

# Technique 19: Word Frequency Analysis
words = ' '.join(df['job']).lower().split()
word_freq = Counter(words)

print("Top 10 Most Common Words:")
print(word_freq.most_common(10))

# Technique 20: One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['job', 'marital', 'education'], drop_first=True)

print("Encoded Dataset Shape:", df_encoded.shape)
df_encoded.head()

