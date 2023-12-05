import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic_df = sns.load_dataset('titanic')

# a. Clean the data by dropping the column with the largest number of missing values
titanic_df_cleaned = titanic_df.dropna(axis=1, thresh=len(titanic_df) * 0.9)

# Display information about the cleaned dataset
print(titanic_df_cleaned.info())

# b. Find total number of passengers with age more than 30
passengers_age_more_than_30 = titanic_df_cleaned[titanic_df_cleaned['age'] > 30].shape[0]
print("Total number of passengers with age more than 30:", passengers_age_more_than_30)

# c. Find total fare paid by passengers of second class
total_fare_second_class = titanic_df_cleaned[titanic_df_cleaned['class'] == 'Second']['fare'].sum()
print("Total fare paid by passengers of second class:", total_fare_second_class)

# d. Compare number of survivors of each passenger class
survivors_by_class = titanic_df_cleaned.groupby('class')['survived'].sum()
print("Number of survivors by passenger class:\n", survivors_by_class)

# e. Compute descriptive statistics for age attribute gender wise
age_statistics_gender_wise = titanic_df_cleaned.groupby('sex')['age'].describe()
print("Descriptive statistics for age attribute gender wise:\n", age_statistics_gender_wise)

# f. Draw a scatter plot for passenger fare paid by Female and Male passengers separately
sns.scatterplot(x='fare', y='sex', data=titanic_df_cleaned)
plt.xlabel('Passenger Fare')
plt.ylabel('Gender')
plt.title('Scatter Plot: Passenger Fare by Gender')
plt.show()

# g. Compare density distribution for features age and passenger fare
sns.kdeplot(titanic_df_cleaned['age'], label='Age', shade=True)
sns.kdeplot(titanic_df_cleaned['fare'], label='Passenger Fare', shade=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Density Distribution: Age vs Passenger Fare')
plt.legend()
plt.show()

# h. Draw the pie chart for three groups labelled as class 1, class 2, class 3 respectively
class_counts = titanic_df_cleaned['class'].value_counts()
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightblue', 'lightgreen'])
plt.title('Distribution of Passenger Classes')
plt.show()

# i. Find % of survived passengers for each class
survival_percentage_by_class = titanic_df_cleaned.groupby('class')['survived'].mean() * 100
print("Percentage of survived passengers for each class:\n", survival_percentage_by_class)