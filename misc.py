from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def word_sense_ambiguitor(context_sentences, target_word):
    """
    Disambiguates the sense of a target word in a list of context sentences using the Naive Bayes algorithm.

    Parameters
    ----------
    context_sentences: list of str
        A list of sentences in which the target word appears.
    target_word: str
        The word whose sense is to be disambiguated.

    Returns
    -------
    str
        The most likely sense of the target word based on the context sentences.
    """
    # Create a list of labels for each context sentence, where each label is the index of the target word in the sentence
    labels = []
    for i, sentence in enumerate(context_sentences):
        labels.append(sentence.index(target_word))
    
    # Create a list of context sentences with the target word replaced by a placeholder
    placeholder = "TARGETWORD"
    context_sentences_modified = []
    for sentence in context_sentences:
        modified_sentence = sentence.replace(target_word, placeholder)
        context_sentences_modified.append(modified_sentence)
    
    # Create a CountVectorizer to convert the context sentences into feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(context_sentences_modified)
    
    # Create a MultinomialNB classifier and fit it to the feature vectors and labels
    clf = MultinomialNB()
    clf.fit(X, labels)
    
    # Get the index of the most likely sense of the target word
    predicted_label = clf.predict(X)[0]
    
    # Return the sense of the target word by looking up the word at the predicted index in the original context sentence
    return context_sentences[predicted_label].split()[predicted_label]


#----------------------------------------------------------------------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

def word_sense_ambiguitor(context_sentences, target_word):
    """
    Disambiguates the sense of a target word in a list of context sentences using the Decision Tree algorithm.

    Parameters
    ----------
    context_sentences: list of str
        A list of sentences in which the target word appears.
    target_word: str
        The word whose sense is to be disambiguated.

    Returns
    -------
    str
        The most likely sense of the target word based on the context sentences.
    """
    # Create a list of labels for each context sentence, where each label is the index of the target word in the sentence
    labels = []
    for i, sentence in enumerate(context_sentences):
        labels.append(sentence.index(target_word))
    
    # Create a list of context sentences with the target word replaced by a placeholder
    placeholder = "TARGETWORD"
    context_sentences_modified = []
    for sentence in context_sentences:
        modified_sentence = sentence.replace(target_word, placeholder)
        context_sentences_modified.append(modified_sentence)
    
    # Create a CountVectorizer to convert the context sentences into feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(context_sentences_modified)
    
    # Create a DecisionTreeClassifier and fit it to the feature vectors and labels
    clf = DecisionTreeClassifier()
    clf.fit(X, labels)
    
    # Get the index of the most likely sense of the target word
    predicted_label = clf.predict(X)[0]
    
    # Return the sense of the target word by looking up the word at the predicted index in the original context sentence
    return context_sentences[predicted_label].split()[predicted_label]
