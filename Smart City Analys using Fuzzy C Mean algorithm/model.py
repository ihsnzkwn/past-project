#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz

# Step 1: Load the dataset from a CSV file
file_path = 'smart_city_dataset.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
df.head(10)


# In[2]:


# Step 2: Data Preprocessing
# (You can customize this based on your dataset)

# Check for missing values
print(df.isnull().sum())

# Handle missing values if any
# For example, you can fill missing values with the mean of the column
df = df.fillna(df.mean())
#df = df.fillna(0)

# Standardize numerical features
scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=[np.number]).columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display the preprocessed data
df.head(10)


# In[3]:


# Step 3: Fuzzy C-Means (FCM) Algorithm

# Extract numerical features for clustering
X = df[numerical_columns].values

# Specify the number of clusters (replace with your desired number)
n_clusters = 10

# Apply Fuzzy C-Means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X.T,
    c=n_clusters,
    m=2,
    error=0.005,
    maxiter=1000,
    init=None
)

# Assign cluster labels to the original DataFrame
df['Cluster'] = np.argmax(u, axis=0)

# Display the DataFrame with cluster assignments
df.head(10)


# In[4]:


#pip install -U scikit-fuzzy


# In[5]:


#pip install flask


# In[6]:


#pip install fuzzy-c-means


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz

# Assuming 'df' is your DataFrame with preprocessed data
# Exclude non-numeric and label columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
scaled_data = StandardScaler().fit_transform(df[numeric_columns])

# Choose the number of clusters (adjusted to match the number of features)
n_clusters = scaled_data.shape[1]

# Apply FCM clustering
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    scaled_data.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
)

# Transpose the cluster centers
cntr = cntr.T

# Print the shapes for diagnostic purposes
print("Shape of scaled_data:", scaled_data.shape)
print("Shape of cluster centers (cntr):", cntr.shape)

# Add cluster labels to the DataFrame
cluster_labels = np.argmax(u, axis=0)
df['Cluster'] = cluster_labels

# Display cluster centers
cluster_centers = pd.DataFrame(cntr, columns=numeric_columns)
print("Cluster Centers:")
print(cluster_centers)


# In[8]:


# Visualization: Scatter plot of two attributes
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Population', y='Area (sq. km)', hue='Cluster', data=df, palette='viridis')
plt.title('Scatter Plot of Clusters (Population vs. Area)')
plt.show()


# In[9]:


print("Shape of cntr:", cntr.shape)
print("Shape of scaled_data:", scaled_data.shape)


# In[10]:


# Assuming 'df' is your DataFrame with preprocessed data and 'Cluster' labels

# Create a DataFrame for each cluster
cluster_0 = df[df['Cluster'] == 0]
cluster_1 = df[df['Cluster'] == 1]
cluster_2 = df[df['Cluster'] == 2]

# Generate descriptions for each cluster based on 'Population' and 'Area (sq. km)'
description_0 = f"Cluster 0: This cluster represents cities with an average population of {cluster_0['Population'].mean():.2f} and an average area of {cluster_0['Area (sq. km)'].mean():.2f} sq. km."
description_1 = f"Cluster 1: This cluster represents cities with an average population of {cluster_1['Population'].mean():.2f} and an average area of {cluster_1['Area (sq. km)'].mean():.2f} sq. km."
description_2 = f"Cluster 2: This cluster represents cities with an average population of {cluster_2['Population'].mean():.2f} and an average area of {cluster_2['Area (sq. km)'].mean():.2f} sq. km."

# Print or return the descriptions
print(description_0)
print(description_1)
print(description_2)


# In[11]:


# Additional cluster descriptions based on different features

# Cluster 0: Smart Infrastructure and Sustainability
description_0_additional = f"Cluster 0 represents cities with a strong focus on smart infrastructure and sustainability. These cities demonstrate high scores in Smart Infrastructure ({cluster_0['Smart Infrastructure Score'].mean():.2f}), Energy Consumption ({cluster_0['Energy Consumption'].mean():.2f}), and Smart Grid Adoption ({cluster_0['Smart Grid Adoption'].mean():.2f}). The emphasis on these factors suggests a commitment to technological advancements and eco-friendly practices."

# Cluster 1: Urban Connectivity and Public Services
description_1_additional = f"Cluster 1 showcases cities with a focus on urban connectivity and public services. The average Public Transport Usage ({cluster_1['Public Transport Usage'].mean():.2f}) is notably high, indicating well-established public transportation systems. Additionally, cities in this cluster tend to have a higher Healthcare Index ({cluster_1['Healthcare Index'].mean():.2f}), emphasizing the importance placed on public services."

# Cluster 2: Environmental Quality and Safety
description_2_additional = f"Cluster 2 is characterized by cities that prioritize environmental quality and safety. Cities in this cluster have a strong emphasis on Air Quality Index ({cluster_2['Air Quality Index'].mean():.2f}) and Safety Index ({cluster_2['Safety Index'].mean():.2f}). The higher values in these indices suggest a focus on creating clean and safe living environments."

# Print or return the additional descriptions
print(description_0_additional)
print(description_1_additional)
print(description_2_additional)


# In[36]:


# Enhanced cluster descriptions with scaled features

# Cluster 0: Smart Infrastructure and Sustainability
description_0_scaled = f"Cluster 0 represents cities with a strong focus on smart infrastructure and sustainability. These cities demonstrate high scores in Smart Infrastructure ({cluster_0['Smart Infrastructure Score'].mean():.2f}), Energy Consumption ({cluster_0['Energy Consumption'].mean():.2f}), and Smart Grid Adoption ({cluster_0['Smart Grid Adoption'].mean():.2f}). The emphasis on these factors, with scaled scores close to 1, suggests a commitment to technological advancements and eco-friendly practices."

# Cluster 1: Urban Connectivity and Public Services
description_1_scaled = f"Cluster 1 showcases cities with a focus on urban connectivity and public services. The average Public Transport Usage ({cluster_1['Public Transport Usage'].mean():.2f}), with a scaled score close to 1, is notably high, indicating well-established public transportation systems. Additionally, cities in this cluster tend to have a higher Healthcare Index ({cluster_1['Healthcare Index'].mean():.2f}), emphasizing the importance placed on public services."

# Cluster 2: Environmental Quality and Safety
description_2_scaled = f"Cluster 2 is characterized by cities that prioritize environmental quality and safety. Cities in this cluster have a strong emphasis on Air Quality Index ({cluster_2['Air Quality Index'].mean():.2f}) and Safety Index ({cluster_2['Safety Index'].mean():.2f}). The higher values in these indices, with scaled scores close to 1, suggest a focus on creating clean and safe living environments."

# Print or return the enhanced descriptions
print(description_0_scaled)
print(description_1_scaled)
print(description_2_scaled)


# In[12]:


# Check the shape of scaled_data
print("Shape of scaled_data:", scaled_data.shape)

# Check the shape of cluster centers (cntr)
print("Shape of cluster centers (cntr):", cntr.shape)


# In[13]:


print(df.columns)


# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Assuming 'df' is your DataFrame
df_numeric = df.drop(['City Name'], axis=1)
X = df_numeric.drop(['Cluster'], axis=1)  # Features
y = df_numeric['Cluster']  # Target variable

# One-hot encode 'City Name'
df_encoded = pd.get_dummies(df, columns=['City Name'], drop_first=True)
X = df_encoded.drop(['Cluster'], axis=1)  # Features
y = df_encoded['Cluster']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Iterate over different values of n_neighbors
for n in range(1, 16):
    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'n_neighbors = {n}, Accuracy: {accuracy}')


# In[15]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Iterate over different values of n_neighbors
for n in range(1, 16):
    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'n_neighbors = {n}, Accuracy: {accuracy}')


# In[16]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Iterate over different values of n_neighbors
for n in range(1, 16):
    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'n_neighbors = {n}, Accuracy: {accuracy}')


# In[17]:


# Example code for trying different values of n_neighbors
for n in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'n_neighbors = {n}, Accuracy: {accuracy}')


# In[18]:


import matplotlib.pyplot as plt

# Lists to store results
n_values = list(range(1, 16))
accuracies = []

# Iterate over different values of n_neighbors
for n in n_values:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(n_values, accuracies, marker='o')
plt.title('Accuracy vs. Number of Neighbors (KNN)')
plt.xlabel('Number of Neighbors (n)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# In[19]:


# Display DataFrame info
print("DataFrame Info:")
print(df.info())


# In[20]:


# Display summary statistics of numeric columns
print("\nSummary Statistics:")
print(df.describe())


# In[21]:


# Display correlation matrix to identify potential features for clustering
print("\nCorrelation Matrix:")
print(df.corr())


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
corr_matrix = df[numeric_columns].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()


# In[23]:


import pandas as pd

# Assuming your DataFrame is named 'df'
description = df.describe()

# Display the description
print(description)


# In[24]:


# Example: Mean of the 'Population' column
population_mean = df['Population'].mean()
print(f"Mean Population: {population_mean}")


# In[25]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from fcmeans import FCM

# Select relevant columns for clustering
columns_for_clustering = ['Public Transport Usage', 'Smart Infrastructure Score', 'Air Quality Index', 'Education Index', 'Healthcare Index', 'Employment Rate']

# Extract the selected columns
data_for_clustering = df[columns_for_clustering]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)

# Apply FCM
n_clusters = 4  # You can adjust this based on your analysis
fcm = FCM(n_clusters=n_clusters)
fcm.fit(scaled_data)

# Get cluster centers and predicted clusters
cluster_centers = fcm.centers
predicted_clusters = fcm.predict(scaled_data)

# Add predicted clusters to the DataFrame
df['Predicted_Cluster'] = predicted_clusters

# Display cluster centers
print("Cluster Centers:")
print(cluster_centers)


# In[26]:


# Display summary statistics with original scale without scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(df.describe())


# In[27]:


pd.reset_option('display.float_format')


# In[28]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('smart_city_dataset.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Correlation matrix
print("\nCorrelation Matrix:")
print(df.corr())


# In[29]:


# Display the column names in your DataFrame
print(df.columns)


# In[30]:


features = ['Population', 'Area (sq. km)']
X = df[features].values


# In[31]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[32]:


fcm = FCM(n_clusters=10)
fcm.fit(X_scaled)


# In[33]:


fuzzy_labels = fcm.predict(X_scaled)


# In[34]:


# Get Cluster Labels
fuzzy_labels = fcm.predict(X_scaled)

# Convert fuzzy labels to hard labels
hard_labels = np.argmax(fuzzy_labels, axis=0)

# Add Cluster Labels to DataFrame
df['Cluster'] = hard_labels


# In[35]:


# Distribution of the target variable (assuming 'Cluster' is your target column)
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', data=df)
plt.title('Distribution of Target Variable')
plt.show()

# Pair plot for numerical features
sns.pairplot(df)
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:




