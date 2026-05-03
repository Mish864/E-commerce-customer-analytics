# %% [5] DATA CLEANING & PREPARATION
import pandas as pd

from sklearn.preprocessing import StandardScaler


file_path = r"C:\Users\ADMIN\OneDrive\Desktop(1)\ecommerce-customer-analytics\E-commerce Customer Behavior - Sheet1.csv"
df = pd.read_csv(file_path)

print(" CHECKING FOR MISSING VALUES ")
print(df.isnull().sum())

if 'Satisfaction Level' in df.columns and df['Satisfaction Level'].isnull().any():
    df['Satisfaction Level'] = df['Satisfaction Level'].fillna(df['Satisfaction Level'].mode().iloc[0])

df.drop_duplicates(inplace=True)

satisfaction_map = {'Unsatisfied': 0, 'Neutral': 1, 'Satisfied': 2}
if 'Satisfaction Level' in df.columns:
    df['Satisfaction_Score'] = df['Satisfaction Level'].map(satisfaction_map)

categorical_cols = ['Gender', 'City', 'Membership Type']
# We only encode columns that actually exist in the dataframe
existing_cats = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=existing_cats, drop_first=True, dtype=int)

if 'Total Spend' in df.columns and 'Items Purchased' in df.columns:
    # Ensure we don't divide by zero
    df['Spend_Per_Item'] = df['Total Spend'] / df['Items Purchased'].replace(0, 1)

    
    spend_threshold = df['Total Spend'].median()
    df['Is_High_Spender'] = (df['Total Spend'] > spend_threshold).astype(int)


scaler = StandardScaler() # FIXED: Now works because of the corrected import


cols_to_scale = ['Total Spend', 'Age', 'Items Purchased', 'Spend_Per_Item']
existing_cols = [col for col in cols_to_scale if col in df.columns]

if existing_cols:
    df[existing_cols] = scaler.fit_transform(df[existing_cols])

df.to_csv("cleaned_customer_data.csv", index=False)
print("\n--- MILESTONE 2 DEBUGGED & COMPLETE ---")
print(df.head())