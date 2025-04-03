import pandas as pd
import seaborn as sns
from smote.oversampling_smote import over_sampling_smote
from method.random_forest import random_forest
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix




# load and preview dataset
dataset_path = "dataset/diabetes.csv"
load_dataset = pd.read_csv(dataset_path)
df = pd.DataFrame(load_dataset)

# drop label class from dataset
X = df.iloc[:,:-1]
y = df['class']

# change class from number to nominal
class_conversion = {
    "class" : {
        0 : "Diabetes",
        1 : "Non Diabetes"
    }
}

df = df.replace(class_conversion)

X = df.iloc[:,:-1]
y = df['class']

# Check Every Classes Count
classes_count = y.value_counts()
sns.countplot(data=df, x="class", label="Count")
L, N = classes_count

# plt.savefig("output/classes_count.png")

# Check Duplicated Datas
duplicated_data = df.duplicated().sum()
df[df.duplicated(keep= False)]

# Check Missing Datas Values
missing_data = df.isnull().sum()
df[df.isnull().any(axis=1)]


X = df.iloc[:,:-1]
y = df["class"]


# data imbalance with oversampling smote
x_smote, y_smote =  over_sampling_smote(X, y)

fig, ax = plt.subplots(1, 2, figsize=(25,10), sharey=True)  # Pastikan 2 subplot

#Original Data Without SMOTE
sns.scatterplot(ax = ax[0], x = df["pres"], y = df["age"], hue = df["class"] )
ax[0].set_title("Original Data Without SMOTE")

#Data With Oversampling SMOTE
sns.scatterplot(ax = ax[1], x = x_smote["pres"], y = x_smote["age"], hue = y_smote)
ax[1].set_title("Data With Oversampling SMOTE")

# plt.savefig("output/smote_data.png")

# test data with random forest method use data with oversampling smote
accuracy, predict = random_forest(x_smote, y_smote)
cm = confusion_matrix(y_smote, predict)

fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(cm, annot=True, fmt='g',cmap='BuGn')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Result Confusion Matrix of Random Forest')
ax.xaxis.set_ticklabels(['Negative', 'Positive'])
ax.yaxis.set_ticklabels([ 'Negative', 'Positive'])


plt.savefig("output/confusion_matrix.png")



if __name__ == "__main__":
    print(f"Diabetes Classes Count : {L}")
    print(f"Non Diabetes Classes Count : {N}")
    print(duplicated_data)
    print(missing_data)
    print(y.value_counts())
    print(y_smote.value_counts())
    print(f"Random forest accuracy : {accuracy}")

