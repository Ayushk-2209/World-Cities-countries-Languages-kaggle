# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


df_city = pd.read_csv('D:\city.csv')
print(df_city.head())

df_country = pd.read_csv('D:\country.csv')
print(df_country.head())

df_language = pd.read_csv('D:\countrylanguage.csv')
print(df_language.head())

# Initial Data Exploration

print(df_city.info())
print(df_country.info())
print(df_language.info())

print(df_city.head())
print(df_city.describe())

# Data Cleaning and Preprocessing for df_country

df_country['IndepYear'] = df_country['IndepYear'].fillna(df_country['IndepYear'].median())
print("clean_country_dataset")
print(df_country.head())

print(df_city.head())

print(df_country.describe())
print(df_country.info())
print(df_country.dtypes)

# Encoding categorical variables

corr_matrix = df_country.corr(numeric_only=True).round(2)
print(corr_matrix)

# Visualizations

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.6f' , linewidths=0.5)
plt.title('Correlation Matrix for df_country')
plt.show()

# Distribution of Life Expectancy and Population

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_country['LifeExpectancy'].dropna(), kde=True , color='lightgreen')
plt.title('Distribution of Life Expectancy')
plt.show()

# Population Distribution

plt.figure(figsize=(10, 10))
sns.histplot(df_country['Population'].dropna(), kde=True , color='blue')
plt.title('Distribution of Population')
plt.tight_layout()
plt.show()

# Population Distribution with Log Scale

plt.figure(figsize=(10, 6))
sns.histplot(df_country['Population'],bins=30 , kde=True)
plt.xscale('log')
plt.title('Distribution of Population (Log Scale)')
plt.xlabel('Population (Log Scale)')
plt.ylabel('count')
plt.show()

# Scatter Plot: Population vs GNP colored by Continent

plt.figure(figsize=(10, 6))
sns.scatterplot(data = df_country,x='Population',y='GNP',hue='Continent')
plt.xscale('log')
plt.yscale('log')   
plt.title('Population vs GNP by Continent (Log Scale)')
plt.xlabel('Population (Log Scale)')
plt.ylabel('GNP (Log Scale)')
plt.show()

# Bar Plot: Average Life Expectancy by Continent

plt.figure(figsize=(10, 6))
sns.barplot(data = df_country,x='Continent',y='LifeExpectancy',ci=None,palette='viridis')
plt.title('Average Life Expectancy by Continent')
plt.xlabel('Continent')
plt.ylabel('Average Life Expectancy')
plt.show()

# Bar Plot: Top 10 Most Populous Countries

df_pop = df_country.nlargest(10, 'Population')
plt.figure(figsize=(12, 6))
sns.barplot(data=df_pop, x='Population', y='Name', palette='magma')
plt.title('Top 10 Most Populous Countries')
plt.xlabel('Population')    
plt.ylabel('Country')
plt.show()

# Correlation Heatmap

plt.figure(figsize=(10, 6))
sns.heatmap(df_country.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix for df_country')
plt.show()

# Box Plot: Life Expectancy by Continent

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_country, x='Continent', y='LifeExpectancy', palette='viridis')
plt.title('Life Expectancy by Continent')
plt.xlabel('Continent')
plt.ylabel('Life Expectancy')
plt.show()

# Box Plots for Numerical Features

num_cols = ['Population', 'LifeExpectancy', 'GNP', 'SurfaceArea']
plt.figure(figsize=(12, 8))
df_country[num_cols].plot(kind='box', subplots=True, layout=(2, 2), figsize=(12, 8), sharex=False, sharey=False)
plt.suptitle('Box Plots of Numerical Features',fontsize=15)
plt.show()