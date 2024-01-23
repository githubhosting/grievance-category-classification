import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
# Replace 'your_file_path.json' with the path to your JSON file
df = pd.read_json('dataset/no_pii_grievance.json')

# Data Cleaning
# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Exploratory Data Analysis
# Basic statistics
print("Basic Statistics:\n", df.describe())

# Count of grievances per state
print("Grievances per State:\n", df['state'].value_counts())

# Visualization
# Distribution of grievances over states
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='state')
plt.title('Distribution of Grievances Across States')
plt.xticks(rotation=45)
plt.show()

# More detailed analysis and visualizations can be added based on specific requirements
