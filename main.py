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
       
        
    print(vectors[:5])
    
        

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
    for word in s.split():
        if word == "$$" or word[0] == "=" or len(word) >= 8 and word[:8] == "interest":
            continue
        
        info = word.split('/')
        vector.append(info[0])
        if len(info) > 1: 
            if info[1] == ".":
                vector.append("PUNCT")
            else:
                vector.append(info[1])        
                
    return vector
    


''' 
Naive Bayes

Source : https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
'''
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def naive_bayes(data, target):
    # Split dataset into training set and test set with 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    
    # Create a Gaussian Classifier
    gnb = GaussianNB()
    
    # Train the model using the training sets
    gnb.fit(X_train, y_train)
    
    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


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
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

''' 
Random Forest

Source : https://www.datacamp.com/tutorial/random-forests-classifier-python
'''
from sklearn.ensemble import RandomForestClassifier
def random_forest(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

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
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


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
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn import datasets
def main():
    #feature_extractor("interest.acl94.txt")
    wine = datasets.load_wine()
    # naive_bayes(wine.data, wine.target)
    # decision_tree(wine.data, wine.target)
    # random_forest(wine.data, wine.target)
    # support_vector_machines(wine.data, wine.target)
    mlp(wine.data, wine.target)
    print("Hello World")
    pass

if __name__ == "__main__":
    main()
    
    
