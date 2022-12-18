'''
Word Sense Disambiguation (WSD) using sklearn
'''

''' 
Feature Extractor
'''
import re
import os
from matplotlib import pyplot as plt
import numpy as np



'''
Function to extract features from a file

Creates a list of feature vecots of windo size 2.

Source : https://practicaldatascience.co.uk/machine-learning/how-to-use-count-vectorization-for-n-gram-analysis
'''
def pre_process_text(text_path, stop_list_path):
    # remove all the stop words from the text of
    stop_list_words = [word for word in open(stop_list_path, 'r').read().split('\n') if word != '']
    text = open(text_path, 'r').read()
    text = re.sub(r"=*", '', text)
    text = re.sub(r"[\[\]]\s", ' ', text)
    for i, word in enumerate(stop_list_words):
        text = re.sub(r"\b" + word.lower() + r"\/\S*\s", '', text, flags=re.IGNORECASE)
        
    new_file = open('processed_text.txt', 'w')
    new_file.write(text)

'''
Function to extract features from a file

Creates a list of feature vecots of windo size 2.

Source : https://practicaldatascience.co.uk/machine-learning/how-to-use-count-vectorization-for-n-gram-analysis
'''
STRING_PATTERN = r"\S+\s*\]?\[?\s+\S+\s*\]?\[?\s+interest[s]?_[0-6]\/NN[S]?\s+\]?\[?\s*\S+\s+\]?\[?\s*\S+"

def feature_extractor(path):
    if not os.path.isfile(path):
            raise Exception('File not found')
    
    file  = open(path, 'r').read()
    regular_expression = re.compile(STRING_PATTERN)
    matches = regular_expression.findall(file)
    
    vectors = []
    
    for match in matches:
        cleaned = remove_miscellaneaous_characters(match)
        v = create_feature_vector(cleaned)
        vectors.append(v)
    
    # returns a list of vectors like this ['declines', 'NNS', 'in', 'IN', 'rates', 'NNS', '.', 'PUNCT']
        
    return vectors
        

'''
Function that takes a string and removes the '[' and ']' characters
'''
def remove_miscellaneaous_characters(s):
    return s.replace('[', '').replace(']', '')

'''
Function that takes a string with the format: 
of/IN  commercial/JJ interest_5/NN  in/IN  china/NP
and returns a list with the format:
['of', 'IN', 'commercial', 'JJ', 'in', 'IN', 'china', 'NP']

'''
def create_feature_vector(s):
    vector = []
    
    # sense corresponds to the sense of the word interest, 
    # we just keep the number ex: 5 -> interest_5
    sense = None
    
    for word in s.split():
        if word.startswith("interests"): 
            sense = int(word[10:11])

            continue
        elif word.startswith("interest"): 
            sense = int(word[9:10])

            continue
        elif word == "$$" or word[0] == "=":
            continue
        
        info = word.split('/')
        vector.append(info[0])
        if len(info) > 1: 
            if info[1] == ".":
                vector.append("PUNCT")
            else:
                vector.append(info[1]) 

    ## TODO: make sure not to take words of last sentence in my vector
    ## TODO: make sure all my vectors are of the same size

    return (sense, vector)
    


''' 
Naive Bayes

Source : https://www.datacamp.com/tutorial/naive-bayes-scikit-learn

To use the Multilabelbinarizer : https://www.kaggle.com/questions-and-answers/66693

'''
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def naive_bayes(splits):
    # Split dataset into training set and test set with 70% training and 30% test
    X_train, X_test, y_train, y_test = splits
    
    # Create a Gaussian Classifier
    gnb = GaussianNB()
    
    ## TODO: utiliser une differente metrique dans le modele pour qu'il optimise cette metrique

    # Train the model using the training sets
    gnb.fit(X_train, y_train)
    
    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of Naive Bayes:", metrics.accuracy_score(y_test, y_pred))

    # Dataset is imbalanced and mutliclassed, so accuracy is not a good metric
    # try precision, recall, f1-score, and confusion matrix
    # try balanced accuracy score

''' 
Decision Tree

Source : https://www.datacamp.com/tutorial/decision-tree-classification-python
'''
from sklearn.tree import DecisionTreeClassifier
def decision_tree(splits):
    X_train, X_test, y_train, y_test = splits
    
    clf = DecisionTreeClassifier()
    
    clf = clf.fit(X_train,y_train)
    
    y_pred = clf.predict(X_test)
    
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of the Decision Tree:",metrics.accuracy_score(y_test, y_pred))

''' 
Random Forest

Source : https://www.datacamp.com/tutorial/random-forests-classifier-python
'''
from sklearn.ensemble import RandomForestClassifier
def random_forest(splits):
    X_train, X_test, y_train, y_test = splits
    
    # Create a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100)
    
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("Accuracy of the Random Forest:",metrics.accuracy_score(y_test, y_pred))

''' 
SVM

Source : https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
'''
from sklearn import svm
def support_vector_machines(splits):
    # Split dataset into training set and test set with 70% training and 30% test
    X_train, X_test, y_train, y_test = splits
    
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel
    
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy of the Support Vector Machines:",metrics.accuracy_score(y_test, y_pred))


''' 
Multilayer Perceptron

Source : https://towardsdatascience.com/deep-neural-multilayer-perceptron-mlp-with-scikit-learn-2698e77155e
'''
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def mlp(splits):
    # Split dataset into training set and test set with 70% training and 30% test
    X_train, X_test, y_train, y_test = splits
    
    param_grid = {
        'hidden_layer_sizes': [(300, 300), (200, 200), (100, 100), (50, 50)], # number of neighbors
    }

    ## Create a mlp Classifier, Je suis pas sur des paramètre tho
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # clf = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=1) # this is already better than previous line
    ## Train the model using the training sets
    # clf.fit(X_train, y_train)

    ## Predict the response for test dataset
    #y_pred = clf.predict(X_test)
    
    mod = GridSearchCV(estimator=MLPClassifier(), param_grid=param_grid, cv=3)

    mod.fit(X_train, y_train)
    print(mod.best_estimator_)

    clf = mod.best_estimator_
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # plot confusion matrix
    plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues, normalize=True)
    plt.show()

    ## Model Accuracy: how often is the classifier correct?
    print("Accuracy of the Multilayer Perceptron:", metrics.accuracy_score(y_test, y_pred))


def plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize=False):
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    classes = model.classes_
    title = 'Confusion matrix'
    # Compute confusion matrix
    #cm = confusion_matrix(y_test, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_test, y_pred)]
    #classes = model.classes_
    if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       print("Normalized confusion matrix")
    else:
       print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from numpy import array

class ManyHotTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    '''
    X is a list of tuples like this : ("string1", "string2", "string3", "string4", "string5", "string6", "string7", "string8")
    we encode every tuple into a list of one-hot encoding like this : 
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ]
    Needs to return either pandas dataframe or numpy array
    '''

    def transform(self, X, y=None):
        transformed_X = array([])
        for data in X:
            data_encoding = array([])
            for word in data:
                #get one-hot encoding of the word
                word_encoding = array([])

                data_encoding.append(word_encoding)
            transformed_X.append(data_encoding)
        return transformed_X


'''
Main function that extract the features and call the classifiers

'''
from sklearn import datasets
from sklearn.preprocessing import MultiLabelBinarizer
def main():
    #vectors = feature_extractor("interest.acl94.txt")
    vectors = feature_extractor("processed_text.txt")

    target, data = list(zip(*vectors))
    
    '''
    We use a multilabelbinarizer to transform the data into matrix intead of a list of strings
    '''

    # create a multilabelbinarizer object
    mlb = MultiLabelBinarizer()
    ## TODO: make a custom transformer/preprocessing that map tuple of words into matrix of one hot vectors
    ## Why? Because mutliLabelBinarizer return a list with 1s and 0s, irrespective of the order of the words
    
    data_2 = mlb.fit_transform(data)

    # splits = train_test_split(data_2, target, test_size=0.3) # UNSTRATIFIED
    # Since we want to compare the different classifiers, we need to split the data in the same way
    # Since the classes are not balanced, we use stratify to make sure the split is done in a balanced way
    # That means that the proportion of each class in the train and test set is the same as in the original dataset
    splits = train_test_split(data_2, target, test_size=0.3, stratify=target) # STRATIFIED
    
    naive_bayes(splits)
    decision_tree(splits)
    random_forest(splits)
    support_vector_machines(splits)
    #mlp(splits)
    pass

if __name__ == "__main__":
    main()
    #pre_process_text("interest.acl94.txt", "stoplist-english.txt")
    
## TODO: tester avec differentes taille de fenetre pour un seul algo et supposez que c'est similaire pour tous

