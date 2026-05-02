import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('cleaned_customer_data.csv')

sns.set_theme(style="whitegrid")

#The Correlation Heatmap 
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['number'])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='RdYlGn', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

#Spending Distribution by High Spender Status 
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x='Total Spend', hue='Is_High_Spender', fill=True)
plt.title('Distribution of Total Spend: High Spenders vs. Others')
plt.show()

#Satisfaction vs. Items Purchased
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Items Purchased', y='Satisfaction_Score', 
                hue='Total Spend', size='Total Spend', palette='viridis')
plt.title('Satisfaction Score vs. Items Purchased')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#Membership Comparison
df['Membership'] = 'Regular'
df.loc[df['Membership Type_Gold'] == 1, 'Membership'] = 'Gold'
df.loc[df['Membership Type_Silver'] == 1, 'Membership'] = 'Silver'

plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='Membership', y='Total Spend', palette='magma')
plt.title('Average Spending by Membership Tier')
plt.show()

#Regional Spending Analysis
cities = ['City_Houston', 'City_Los Angeles', 'City_Miami', 'City_New York', 'City_San Francisco']
city_spend = [df[df[city] == 1]['Total Spend'].mean() for city in cities]
city_names = [city.replace('City_', '') for city in cities]

plt.figure(figsize=(10, 6))
plt.bar(city_names, city_spend, color='skyblue')
plt.title('Average Total Spend per City')
plt.ylabel('Average Spend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()