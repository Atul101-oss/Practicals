import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import norm
from scipy import stats

# a. Load data into pandas' data frame.
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Display information about the dataset
print(iris_df.info())

# b. Find the number of missing values in each column
missing_values = iris_df.isnull().sum()
print("Number of missing values in each column:\n", missing_values)

# c. Plot bar chart to show the frequency of each class label
sns.countplot(x='target', data=iris_df)
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.title('Frequency of Each Class Label')
plt.show()

# d. Draw a scatter plot for Petal Length vs Sepal Length and fit a regression line
sns.regplot(x='sepal length (cm)', y='petal length (cm)', data=iris_df)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Scatter Plot: Petal Length vs Sepal Length')
plt.show()

# e. Plot density distribution for feature Petal width
sns.histplot(iris_df['petal width (cm)'], kde=True, color='blue', bins=20)
plt.xlabel('Petal Width (cm)')
plt.ylabel('Density')
plt.title('Density Distribution of Petal Width')
plt.show()

# f. Use a pair plot to show pairwise bivariate distribution in the Iris Dataset
sns.pairplot(iris_df, hue='target')
plt.suptitle('Pairwise Bivariate Distribution in Iris Dataset')
plt.show()

# g. Draw heatmap for any two numeric attributes
numeric_attributes = iris_df.select_dtypes(include='number')
sns.heatmap(numeric_attributes.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap for Numeric Attributes')
plt.show()

# h. Compute mean, mode, median, standard deviation, confidence interval, and standard error
numeric_stats = iris_df.describe().transpose()
print("Numeric Features Statistics:\n", numeric_stats)

# i. Compute correlation coefficients between each pair of features and plot heatmap
correlation_matrix = iris_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Coefficients Heatmap')
plt.show()