# %%
import pandas as pd
print("milestone 1 setup working")

# %%
import pandas as pd

df = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop(1)\\ecommerce-customer-analytics\\E-commerce Customer Behavior - Sheet1.csv")

print(df.head())
print(df.info())
print(df.describe())
df

# %%
stats = df[['Total Spend', 'Items Purchased', 'Average Rating']].describe()
print(stats)

satisfaction_counts = df['Satisfaction Level'].value_counts()
print(satisfaction_counts)

# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set up the figure for 4 visuals (2x2 grid)
plt.figure(figsize=(15, 12))
sns.set_theme(style="whitegrid")

# Visual 1: Distribution of Total Spend (Ratio Data)
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop(1)\\ecommerce-customer-analytics\\E-commerce Customer Behavior - Sheet1.csv")

plt.figure(figsize=(15, 12))
sns.set_theme(style="whitegrid")

plt.subplot(2, 2, 1)
sns.histplot(df['Total Spend'], kde=True, color='skyblue')
plt.title('1. Distribution of Total Spend', fontsize=14)
plt.xlabel('Spend Amount')

plt.subplot(2, 2, 2)
membership_counts = df['Membership Type'].value_counts()
plt.pie(membership_counts, labels=membership_counts.index, autopct='%1.1f%%',
        colors=['#ffd700', '#c0c0c0', '#cd7f32'], startangle=140)
plt.title('2. Membership Tier Distribution', fontsize=14)

plt.subplot(2, 2, 3)
sns.countplot(x='Satisfaction Level', data=df, palette='viridis',
              order=['Satisfied', 'Neutral', 'Unsatisfied'])
plt.title('3. Customer Satisfaction Levels', fontsize=14)
plt.ylabel('Number of Customers')

plt.subplot(2, 2, 4)
sns.scatterplot(x='Items Purchased', y='Average Rating', hue='Gender', data=df, alpha=0.7)
plt.title('4. Items Purchased vs. Average Rating', fontsize=14)

plt.tight_layout()
plt.show()
sns.histplot(df['Total Spend'], kde=True, color='skyblue')
plt.title('1. Distribution of Total Spend', fontsize=14)
plt.xlabel('Spend Amount')

# Visual 2: Membership Type Proportions (Ordinal Data)
plt.subplot(2, 2, 2)
membership_counts = df['Membership Type'].value_counts()
plt.pie(membership_counts, labels=membership_counts.index, autopct='%1.1f%%', 
        colors=['#ffd700', '#c0c0c0', '#cd7f32'], startangle=140)
plt.title('2. Membership Tier Distribution', fontsize=14)

# Visual 3: Satisfaction Level Frequency (Categorical Data)
plt.subplot(2, 2, 3)
sns.countplot(x='Satisfaction Level', data=df, palette='viridis', order=['Satisfied', 'Neutral', 'Unsatisfied'])
plt.title('3. Customer Satisfaction Levels', fontsize=14)
plt.ylabel('Number of Customers')

# Visual 4: Items Purchased vs. Average Rating (Relationship)
plt.subplot(2, 2, 4)
sns.scatterplot(x='Items Purchased', y='Average Rating', hue='Gender', data=df, alpha=0.7)
plt.title('4. Items Purchased vs. Average Rating', fontsize=14)

# Final formatting
plt.tight_layout()
plt.show()

# %%



