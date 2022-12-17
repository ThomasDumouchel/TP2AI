'''
Pour déterminer le sens du mot, on utilise les informations dans son contexte. Les informations contextuelles doivent être 
extraites au préalable (ceci n’est pas fait par Scikit-learn). Il y a plusieurs types d’informations contextuelles. Voici 
deux types d’information contextuelle :

-   L’ensemble des mots avant et les mots après (dans un sac de mots, sans ordre). Dans le premier exemple proposés ci-dessus, 
    si on retient les 2 mots avant et 2 mots après (le mot interest) sont {declines, in, rate, .}.
-   Les catégories des mots autour. Pour les mêmes 4 mots du premier exemple, nous allons avoir : «NNS », «IN », «NNS », «. » 
    (la ponctuation). 

Ces catégories sont généralement prises en compte en ordre (C-2=NNS, C-1=IN, C1=NNS, et C2=.) afin de tenir compte de la structure syntaxique.
Ces deux groupes de caractéristiques sont ceux que vous devez utiliser au minimum.


Le texte annoté contient le résultat d’une analyse de partie-de-discours (part-of- speech) + annotation de sens du mot interest. Voici un exemple :
[ yields/NNS ] on/IN [ money-market/JJ mutual/JJ funds/NNS ] continued/VBD to/TO slide/VB ,/, amid/IN [ signs/NNS ] that/IN [ portfolio/NN managers/NNS ] 
expect/VBP [ further/JJ declines/NNS ] in/IN [ interest_6/NN rates/NNS ] ./.
$$
[ longer/JJR maturities/NNS ] are/VBP thought/VBN to/TO indicate/VB [ declining/VBG interest_6/NN rates/NNS ] because/IN [ they/PP ] permit/VBP 
[ portfolio/NN managers/NNS ] to/TO retain/VB relatively/RB [ higher/JJR rates/NNS ] for/IN [ a/DT longer/JJR period/NN ] ./.
Dans cet exemple, les crochets [ ] enferment un groupe nominal. Chaque mot est suivi de sa catégorie grammaticale (e.g. /NNS), et le mot ambigu, interest, 
est annotéde son sens (_6, c’est-à-dire le 6ième sens). Les ponctuations sont elles-mêmes leur propre catégorie (comme dans ./. àla fin d’une phrase). 
Les phrases sont séparées par une ligne de $$.
'''
import os

class lineIterator(object):
    def __init__(self, line):
        self.lineWords = line.split()
        self.pointer = 0
        self._next = self.findNext()

    def findNext(self):
        while self.pointer < len(self.lineWords) and self.lineWords[self.pointer] in {'[', ']', '======================================'}:
            self.pointer += 1
        if self.pointer < len(self.lineWords):
            return self.lineWords[self.pointer]
        return None

    def next(self):
        ret = (self.pointer, self._next)
        self.pointer += 1
        self._next = self.findNext()
        return ret

    def hasNext(self):
        return self._next != None

class TP2Extractor(object):
    def __init__(self):
        self.__wordX = []
        self.__groupX = []
        self.__y = []
    
    def getWordX(self):
        return self.__wordX
    
    def getGroupX(self):
        return self.__groupX

    def getY(self):
        return self.__y

    def extractWordAndGroup(self, word):
        wordAndGroup = word.split('/')
        # It's possible that the word has no group, in which case we put an empty string for the group
        if len(wordAndGroup) == 1:
            wordAndGroup.append("")
        # If the word has many /, we have an error:
        if len(wordAndGroup) > 2:
            raise Exception('Error in the file. The wordGroup ' + word + ' has more than one /')
        return wordAndGroup

    def extract(self, annotatedTextFilePath):
        '''
        I assume only one word interest in a sentence. It is the case.
        What to do if the word interest doesn't have two words before or two words after? What should we put in the word bag instead? Or should we put nothing?
        ??? I'll put ["", ""], meaning empty string.
        Retourne
        X: une liste de listes de mots. Ce seront les 2 mots avant et les 2 mots après le mot interest.
        y: une liste de sens du mot interest.
        pour tout i, X[i] seront les caractéristiques du mot interest de la phrase i, et y[i] sera l'étiquette du mot interest de la phrase i.
        '''
        # verify that the file exists
        if not os.path.isfile(annotatedTextFilePath):
            raise Exception('File not found')
        with open(annotatedTextFilePath, 'r') as f:
            lastTwo = ["",""]
            for line in f:
                if line == '$$\n':
                    # whenever we encounter a $$, we have a new sentence
                    lastTwo = ["",""]
                    continue
                interestFound = False
                lineIter = lineIterator(line)
                while lineIter.hasNext():
                    index, word = lineIter.next()
                    if word.startswith('interest_') or word.startswith('interests_'):
                        interestFound = True
                        # we just encountered the word interest
                        # we need to get the next two words
                        nextTwo = []
                        while len(nextTwo) < 2: 
                            if lineIter.hasNext():
                                nextIndex, NextWord = lineIter.next()
                                nextTwo.append(NextWord)
                            else:
                                nextTwo.append("")
                        # combine lastTwo and nextTwo to get the word bag, and add it to X, and the label to y
                        wContext = []
                        gContext = []
                        for wg in lastTwo + nextTwo:
                            w, g = self.extractWordAndGroup(wg)
                            wContext.append(w)
                            gContext.append(g)
                        self.__wordX.append(wContext)
                        self.__groupX.append(gContext)
                        self.__y.append(word[word.index("_") + 1])
                    else:
                        lastTwo.pop(0)
                        lastTwo.append(word)
                if not interestFound:
                    # we didn't find the word interest in this sentence
                    # for debugging purposes
                    print('interest not found in sentence: ' + line)

ex = TP2Extractor()
ex.extract('interest.acl94.txt')
print(ex.getGroupX()[:3])
print(ex.getWordX()[:3])