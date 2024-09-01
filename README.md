```markdown
# Data Visualization Project

This project involves analyzing and visualizing data from a space travel dataset to understand various attributes related to passengers and their journey.

## Overview

The dataset contains information about passengers, including their home planet, age, and various services used during their journey. The goal is to clean, preprocess, and visualize the data to gain insights.

## Libraries Used

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.

## Dataset

The dataset used in this project is `train.csv`, which contains the following columns:

| Column          | Description                                   |
|------------------|-----------------------------------------------|
| PassengerId      | Unique identifier for each passenger         |
| HomePlanet       | The planet the passenger is from             |
| CryoSleep        | Indicates if the passenger is in cryo sleep  |
| Cabin            | Cabin number of the passenger                 |
| Destination      | The destination of the passenger              |
| Age              | Age of the passenger                          |
| VIP              | Indicates if the passenger is a VIP          |
| RoomService      | Amount spent on room service                  |
| FoodCourt        | Amount spent in the food court                |
| ShoppingMall     | Amount spent in the shopping mall             |
| Spa              | Amount spent on spa services                  |
| VRDeck           | Amount spent on VR Deck services              |
| Name             | Name of the passenger                         |
| Transported      | Indicates if the passenger was transported    |

## Data Exploration

### Loading the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
print(df.head())
```

### Data Summary

The following summary provides insights into the dataset:

```python
print(df.describe())
print(df.info())
```

### Unique Values

We can check the unique values for certain columns:

```python
print(df['Destination'].unique())
print(df['HomePlanet'].unique())
```

### Null Values

The dataset contains some null values that need to be addressed. We found the following columns with missing data:

- CryoSleep
- Cabin
- Age
- VIP
- RoomService
- FoodCourt
- ShoppingMall
- Spa
- VRDeck

### Feature Engineering

We added a new feature, `HomeDestination`, by combining `HomePlanet` and `Destination`. The updated dataframe is:

```python
df['HomeDestination'] = df['HomePlanet'] + '_' + df['Destination']
df = pd.get_dummies(df, columns=['HomeDestination'], drop_first=True)
```

### Data Cleaning

We dropped unnecessary columns and filled null values using various strategies:

```python
df = df.drop(['Name', 'HomePlanet', 'Destination'], axis='columns')

df['CryoSleep'].fillna(False, inplace=True)
df['VIP'].fillna(True, inplace=True)

# KNN Imputation for other columns
from sklearn.impute import KNNImputer

knn_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
imputer = KNNImputer(n_neighbors=5, weights='uniform')
df[knn_columns] = pd.DataFrame(imputer.fit_transform(df[knn_columns]), columns=knn_columns)
```

## Visualization

After cleaning and preprocessing the data, we can visualize various aspects of the dataset to gain insights. Use `matplotlib` to create plots as needed.

## Conclusion

This project aims to provide insights into the space travel dataset by exploring passenger data, cleaning the dataset, and visualizing key attributes. Further analysis can be done to enhance understanding and derive actionable insights.
