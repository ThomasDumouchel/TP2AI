# import naive bayes classifier from sklearn
from sklearn.naive_bayes import GaussianNB
# import word bag extractor from ContextExtractor.py
from TP2Extractor import TP2Extractor
import pandas as pd

ex = TP2Extractor()
ex.extract('interest.acl94.txt')

# convert X to a pandas dataframe
wordX = pd.DataFrame(ex.getWordX())         # wordX, c'est nos vecteurs d'attributs avant deux mots avant et après le mot interest
groupX = pd.DataFrame(ex.getGroupX())       # groupX, c'est nos vecteurs d'attributs après le groupe des deux mots avant et après le mot interest
y = pd.DataFrame(ex.getY())                 # y, c'est notre étiquette, donc le sens du mot interest

print(wordX.shape())


# # create a Gaussian Naive Bayes classifier
# clf = GaussianNB()

# # train the classifier
# clf.fit(wordX, y)

# # test the classifier
# print(clf.predict([['interest', 'NN', 'is', 'VBZ'], ['interest', 'NN', 'is', 'VBZ']]))


