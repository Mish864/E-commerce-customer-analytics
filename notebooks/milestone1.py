# %% [1] SETUP & DATA LOADING
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:\\Users\\ADMIN\\OneDrive\\Desktop(1)\\ecommerce-customer-analytics\\E-commerce Customer Behavior - Sheet1.csv"
df = pd.read_csv(file_path)

# %% [2] DATA SCHEMA (Metadata)
# This displays the column names, data types, and non-null counts
print("--- DATASET SCHEMA ---")
print(df.info())

# %% [3] DESCRIPTIVE STATISTICS
print("\n--- NUMERICAL SUMMARY ---")
# .T (transpose) makes the statistics easier to read as a list
print(df.describe().T)

print("\n--- CATEGORICAL SUMMARY ---")
# This summarizes non-numeric columns like Gender, City, and Satisfaction Level
print(df.describe(include=['object']).T)

print("\n--- SATISFACTION VALUE COUNTS ---")
print(df['Satisfaction Level'].value_counts())

# %% [4] DATA VISUALIZATION
# Set global style
sns.set_theme(style="whitegrid")

# Create a 2x2 grid for the plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Visual 1: Distribution of Total Spend (Histplot)
sns.histplot(df['Total Spend'], kde=True, color='skyblue', ax=axes[0, 0])
axes[0, 0].set_title('1. Distribution of Total Spend', fontsize=14)
axes[0, 0].set_xlabel('Spend Amount')

# Visual 2: Membership Tier Proportions (Pie Chart)
membership_counts = df['Membership Type'].value_counts()
axes[0, 1].pie(membership_counts, labels=membership_counts.index, autopct='%1.1f%%',
              colors=['#ffd700', '#c0c0c0', '#cd7f32'], startangle=140)
axes[0, 1].set_title('2. Membership Tier Distribution', fontsize=14)

# Visual 3: Satisfaction Level Frequency (Countplot)
sns.countplot(x='Satisfaction Level', data=df, palette='viridis',
              order=['Satisfied', 'Neutral', 'Unsatisfied'], ax=axes[1, 0])
axes[1, 0].set_title('3. Customer Satisfaction Levels', fontsize=14)
axes[1, 0].set_ylabel('Number of Customers')

# Visual 4: Relationship between Items and Rating (Scatterplot)
sns.scatterplot(x='Items Purchased', y='Average Rating', hue='Gender', data=df, alpha=0.7, ax=axes[1, 1])
axes[1, 1].set_title('4. Items Purchased vs. Average Rating', fontsize=14)

# Final formatting adjustments
plt.tight_layout()
plt.show()