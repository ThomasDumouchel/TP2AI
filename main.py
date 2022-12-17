'''
Word Sense Disambiguation (WSD) using sklearn
'''

''' 
Feature Extractor
'''
import re
import os

from sklearn.model_selection import train_test_split

STRING_PATTERN = r"\S+\s*\]?\[?\s+\S+\s*\]?\[?\s+interest_[0-6]\/NN\s+\]?\[?\s*\S+\s+\]?\[?\s*\S+"

'''
Function to extract features from a file

Creates a list of feature vecots of windo size 2.

Source : https://practicaldatascience.co.uk/machine-learning/how-to-use-count-vectorization-for-n-gram-analysis
'''
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
        if len(word) >= 8 and word[:8] == "interest": 
            sense = int(word[9:10])
            continue
        if word == "$$" or word[0] == "=":
            continue
        
        info = word.split('/')
        vector.append(info[0])
        if len(info) > 1: 
            if info[1] == ".":
                vector.append("PUNCT")
            else:
                vector.append(info[1])        
                
    return (sense, vector)
    


''' 
Naive Bayes

Source : https://www.datacamp.com/tutorial/naive-bayes-scikit-learn

To use the Multilabelbinarizer : https://www.kaggle.com/questions-and-answers/66693

'''
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def naive_bayes(data, target):
    # Split dataset into training set and test set with 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    
    # Create a Gaussian Classifier
    gnb = GaussianNB()
    
    # Train the model using the training sets
    print(y_train[0])
    gnb.fit(X_train, y_train)
    
    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy of Naive Bayes:",metrics.accuracy_score(y_test, y_pred))


''' 
Decision Tree

Source : https://www.datacamp.com/tutorial/decision-tree-classification-python
'''
from sklearn.tree import DecisionTreeClassifier
def decision_tree(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    
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
def random_forest(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    
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
def support_vector_machines(data, target):
    # Split dataset into training set and test set with 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    
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
def mlp(data, target):
    # Split dataset into training set and test set with 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    
    # Create a mlp Classifier, Je suis pas sur des paramètre tho
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    
    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy of the Multilayer Perceptron:",metrics.accuracy_score(y_test, y_pred))

'''
Main function that extract the features and call the classifiers

'''
from sklearn import datasets
from sklearn.preprocessing import MultiLabelBinarizer
def main():
    vectors = feature_extractor("interest.acl94.txt")
    target, data = list(zip(*vectors))
    
    
    '''
    We use a multilabelbinarizer to transform the data into matrix intead of a list of strings
    '''
    # create a multilabelbinarizer object
    mlb = MultiLabelBinarizer()
    
    data_2 = mlb.fit_transform(data)

    # target_2 = mlb.fit_transform(target)
    # print(data_2[0])
    # print(target_2[0])
    
    # wine = datasets.load_wine()
    
    # [[] []]
    # print(wine.target[:])
    
    naive_bayes(data_2, target)
    decision_tree(data_2, target)
    random_forest(data_2, target)
    support_vector_machines(data_2, target)
    # mlp(wine.data, wine.target)
    print("Hello World")
    pass

if __name__ == "__main__":
    main()
    
    
