import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('C:\\Users\\ADMIN\\OneDrive\\Desktop(1)\\ecommerce-customer-analytics\\cleaned_customer_data.csv')
df['Spend_Per_Item'] = df['Total Spend'] / df['Items Purchased']
    
plt.figure(figsize=(16, 10))
sns.set_style("white") 

    # Visual 1: Violin Plot (Distribution & Density)
plt.subplot(2, 2, 1)
sns.violinplot(x='Membership Type', y='Total Spend', data=df, palette="muted")
plt.title('1. Spending Density by Membership')

    # Visual 2: Heatmap (Regional Satisfaction)
plt.subplot(2, 2, 2)
ct = pd.crosstab(df['City'], df['Satisfaction Level'])
sns.heatmap(ct, annot=True, cmap="YlGnBu", cbar=False)
plt.title('2. Satisfaction Concentration by City')

    # Visual 3: Bubble Chart (Multidimensional)
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='Age', y='Total Spend', hue='Satisfaction Level', size='Items Purchased', alpha=0.7)
plt.title('3. Age vs Spend (Sized by Items)')

    # Visual 4: Correlation Heatmap
plt.subplot(2, 2, 4)
corr = df[['Total Spend', 'Items Purchased', 'Average Rating', 'Spend_Per_Item']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title('4. Feature Correlation Matrix')

plt.tight_layout()
plt.show()

