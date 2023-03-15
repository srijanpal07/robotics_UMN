import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from MyDiscriminant import GaussianDiscriminant

# load data
df = np.genfromtxt("training_data.txt", delimiter=",")
dftest = np.genfromtxt("test_data.txt", delimiter=",")
Xtrain = df[:, 0:8]
ytrain = df[:, 8]
Xtest = dftest[:, 0:8]
ytest = dftest[:, 8]

# define the model with a Gaussian Discriminant function (class-dependent covariance)
clf = GaussianDiscriminant(2, 8, [0.3, 0.7])

# update the model based on training data
clf.fit(Xtrain, ytrain)

# evaluate on test data
predictions = clf.predict(Xtest)

print(
    "Confusion Matrix for Gaussian Discriminant with class-dependent covariance"
)

conf_matrix = confusion_matrix(predictions, ytest)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(
            x=j,
            y=i,
            s=conf_matrix[i, j],
            va="center",
            ha="center",
            size="xx-large",
        )
fig.savefig("confusion_matrix.png")
plt.show()


# define the model with a Gaussian Discriminant function (class-independent covariance)
clf = GaussianDiscriminant(2, 8, [0.3, 0.7], shared_cov=True)

# update the model based on training data
clf.fit(Xtrain, ytrain)

# evaluate on test data
predictions = clf.predict(Xtest)

# evaluate on test data
print(
    "Confusion Matrix for Gaussian Discriminant with class-independent covariance"
)
conf_matrix = confusion_matrix(predictions, ytest)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(
            x=j,
            y=i,
            s=conf_matrix[i, j],
            va="center",
            ha="center",
            size="xx-large",
        )
fig.savefig("confusion_matrix_shared.png")
plt.show()
