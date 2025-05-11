import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()

# Convert the dataset into a pandas DataFrame
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target (species) column to the DataFrame
data['species'] = iris.target

# Map the numeric target values to species names
data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first few rows
print(data.head())

# To Check the data types of each column
print(data.info())

# To Check for missing values
print(data.isnull().sum())

# to Get basic descriptive statistics
print(data.describe())



data.fillna(data.mean(), inplace=True)

# Calculate mean, median, and standard deviation
mean_values = data.mean()
median_values = data.median()
std_values = data.std()

print("Mean Values:\n", mean_values)
print("Median Values:\n", median_values)
print("Standard Deviation Values:\n", std_values)


# Group by species and calculate the mean of each numerical column
grouped_by_species = data.groupby('species').mean()
print(grouped_by_species)


import matplotlib.pyplot as plt
import seaborn as sns

# Create a line plot for petal length by species
plt.figure(figsize=(10, 6))
sns.lineplot(x="species", y="petal length (cm)", data=data)
plt.title('Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()


# Create a bar chart for average petal length by species
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='petal length (cm)', data=data)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()


# this Create a histogram for petal length
plt.figure(figsize=(10, 6))
sns.histplot(data['petal length (cm)'], bins=20, kde=True, color='purple')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()


# this Create a scatter plot for sepal length vs. petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=data)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()


try:
    data = pd.read_csv('your_dataset.csv')
except FileNotFoundError:
    print("The dataset file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
