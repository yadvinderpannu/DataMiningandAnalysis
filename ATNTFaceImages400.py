import numpy as np
from sklearn import svm, model_selection


def knn_classifier(X, Y, Xtest, Ytest, k=3):

    Youtput = []
    for xt in np.transpose(Xtest):
        euclidean = np.linalg.norm(X - np.vstack(xt), axis=0)
        neighbors = Y[np.argsort(euclidean)[:k]]
        Youtput.append(np.argmax(np.bincount(np.array(neighbors,dtype=np.int32))))

    score = np.mean(np.array(Youtput) == Ytest)
    return score

def centroid_classifier(X, Y, Xtest, Ytest):

    labels = np.unique(Y)
    centroids = np.zeros((X.shape[0],len(labels)))
    Youtput = []

    for l in range(len(labels)):
        centroids[:,l] = np.mean(X[:,Y == labels[l]],axis=1)

    for xt in np.transpose(Xtest):
        distance = np.linalg.norm(centroids - np.vstack(xt), axis=0)
        Youtput.append(labels[np.argmin(distance)])

    score = np.mean(np.array(Youtput) == Ytest)
    return score



def linear_classification(X, Y, Xtest, Ytest):

    labels = np.unique(Y)
    Ymat = np.zeros((labels.shape[0], Y.shape[0]))
    for l in range(labels.shape[0]):
        Ymat[l,np.where(Y == labels[l])[0]] = 1

    res = np.dot(np.linalg.pinv(np.transpose(X)),np.transpose(Ymat))
    Youtput = np.dot(np.transpose(res), Xtest)

    Youtput = np.argmax(Youtput, axis=0)
    Youtput = labels[Youtput]
    score = np.mean(Youtput == Ytest)
    return score



def svm_classifier(X, Y, Xtest, Ytest):

    model = svm.SVC(kernel='linear', C=1, gamma=1)
    model.fit(np.transpose(X), Y)
    score = model.score(np.transpose(Xtest), Ytest)
    return score


# main execution
filename = "ATNTFaceImages400.txt"
dataset = np.loadtxt(filename, delimiter=",")
X = dataset[1:] 
Y = dataset[0]

kfold_cv = model_selection.StratifiedKFold(n_splits=5)
kfold_cv.get_n_splits(np.transpose(X), Y)

linear_score = []
knn_score = []
centroid_score = []
svm_score = []

for train, test in kfold_cv.split(np.transpose(X), Y):
    Xtrain = X[:,train]
    Xtest = X[:,test]
    Ytrain = Y[train]
    Ytest = Y[test]

    knn_score.append(knn_classifier(Xtrain, Ytrain, Xtest, Ytest))
    centroid_score.append(centroid_classifier(Xtrain, Ytrain, Xtest, Ytest))
    linear_score.append(linear_classification(Xtrain, Ytrain, Xtest, Ytest))
    svm_score.append(svm_classifier(Xtrain, Ytrain, Xtest, Ytest))

print("kNN classifier for k= 3")
for i in range(len(knn_score)):
    print("K = %d, accuracy = %5.3f" %(i+1, knn_score[i]))

print("\nCentroid classifier")
for i in range(len(centroid_score)):
    print("K = %d, accuracy = %5.3f" %(i+1, centroid_score[i]))

print("\nLinear regression classifier")
for i in range(len(linear_score)):
    print("K = %d, accuracy = %5.3f" %(i+1, linear_score[i]))

print("\nSVM classifier")
for i in range(len(svm_score)):
    print("K = %d, accuracy = %5.3f" %(i+1, svm_score[i]))

# yadvinderpannu
