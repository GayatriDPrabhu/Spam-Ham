import math
import os
import operator
import sys

classValues = {'ham': 1.0, 'spam': 0.0}
docCount = {'ham': 0.0, 'spam': 0.0}

def getWords(filePath):
    words = []
    try:
       with open(filePath) as f:
           words = [word for line in f for word in line.split()]
    except OSError as ex:
        print(ex.message)
    finally:
        return words

def getStopWordsList(fileName):
    fileDirPath = os.getcwd() + '/' + fileName
    words = getWords(fileDirPath)
    return words

def getData():
    data = {'ham': [], 'spam': []}
    for fileName in os.listdir(os.getcwd() + '/train/ham'):
        words = getWords(os.getcwd() + '/train/ham/' + fileName)
        if len(words) > 0:
            data['ham'].append(words)
        docCount['ham'] += 1.0
    for fileName in os.listdir(os.getcwd() + '/train/spam'):
        words = getWords(os.getcwd() + '/train/spam/' + fileName)
        if len(words) > 0:
            data['spam'].append(words)
        docCount['spam'] += 1.0
    return data

def buildVocab(data, skipWordsList):
    vocab = []
    for classType in data.iterkeys():
        for item in data[classType]:
            for word in item:
                if word not in vocab and word.lower() not in skipWordsList:
                    vocab.append(word)
    return vocab

def featureSelection(data, vocab):
    totalDocCount = sum(docCount.values())
    newVocab = {}
    for word in vocab:
        wordDocCount = {'ham': 0.0, 'spam': 0.0}
        for classType in data:
            for doc in data[classType]:
                if word in doc:
                    wordDocCount[classType] += 1.0
        wordPresentCount = sum(wordDocCount.values())
        wordAbsentCount = totalDocCount - wordPresentCount + 1
        term1 = wordDocCount['ham'] / totalDocCount
        if wordPresentCount == 0:
            term2 = 1
        else:
            term2 = ((totalDocCount * wordDocCount['ham']) / (wordPresentCount * docCount['ham']))
            if term2 == 0:
                term2 = 1
        term3 = (docCount['ham'] - wordDocCount['ham']) / totalDocCount
        if wordAbsentCount == 0:
            term4 = 1
        else:
            term4 = (totalDocCount * (docCount['ham'] - wordDocCount['ham'])) / (wordAbsentCount * docCount['ham'])
            if term4 == 0:
                term4 = 1
        term5 = wordDocCount['spam'] / totalDocCount
        if wordPresentCount == 0:
            term6 = 1
        else:
            term6 = (totalDocCount * wordDocCount['spam']) / (wordPresentCount * docCount['spam'])
            if term6 == 0:
                term6 = 1
        term7 = (docCount['spam'] - wordDocCount['spam']) / totalDocCount
        if wordAbsentCount == 0:
            term8 = 1
        else:
            term8 = (totalDocCount * (docCount['spam'] - wordDocCount['spam'])) / (wordAbsentCount * docCount['spam'])
            if term8 == 0:
                term8 = 1
        newVocab[word] = term1 * math.log(term2, 2)
        newVocab[word] += term3 * math.log(term4, 2)
        newVocab[word] += term5 * math.log(term6, 2)
        newVocab[word] += term7 * math.log(term8, 2)
    sortedList = sorted(newVocab.items(), key=operator.itemgetter(1), reverse=True)
    newVocab = []
    for i in range(0, k):
        newVocab.append(sortedList[i][0])
    return newVocab

def getClassText(docs):
    classText = []
    for docWords in docs:
        for word in docWords:
            classText.append(word)
    return classText

def getFileClass(filePath, vocab, prior, condProb):
    words = getWords(filePath)
    classScore = {'ham': 0.0, 'spam': 0.0}
    for classType in classScore.keys():
        classScore[classType] = math.log(prior[classType])
        for word in words:
            if word in condProb[classType]:
                classScore[classType] += math.log(condProb[classType][word])
    return classScore.keys()[classScore.values().index(max(classScore.values()))]

def trainBN(data, vocab):
    prior = {}
    condProb = {}
    for classType in docCount.keys():
        condProb[classType] = {}
        prior[classType] = docCount[classType] / sum(docCount.values())
        classText = getClassText(data[classType])
        wordCount = {}
        for word in vocab:
            wordCount[word] = classText.count(word) * 1.0
        for word in vocab:
            condProb[classType][word] = (wordCount[word] + 1.0) / (sum(wordCount.values()) + len(wordCount.values()))
    return prior, condProb

def testBN(vocab, prior, condProb):
    accuracy = {'ham': 0.0, 'spam': 0.0}
    totalSize = 0.0
    for classType in accuracy.keys(): 
        for fileName in os.listdir(os.getcwd() + '/test/' + classType):
            fileClass = getFileClass(os.getcwd() + '/test/' + classType + '/' + fileName, vocab, prior, condProb)
            if fileClass == classType:
                accuracy[classType] += 1.0
            totalSize += 1.0
    return (sum(accuracy.values()) / totalSize) * 100

def initializeWeights(vocab):
    wts = {'bias': 0.0}
    for word in vocab:
        wts[word] = 0.0
    return wts

def getFeatures(doc):
    features = {'bias': 1.0}
    for word in doc:
        features[word] = doc.count(word)
    return features

def getFileFeatures(fileDir, filename):
    features = {'bias': 1.0}
    words = getWords(fileDir + '/' + filename)
    for word in words:
        features[word] = words.count(word)
    return features

def getWeightedSum(features, wts):
    weightedSum = 0.0
    for feature, value in features.items():
        if feature in wts:
            weightedSum += value * wts[feature]
    return weightedSum

def getClassProbability(features, wts):
    weightedSum = getWeightedSum(features, wts)
    try:
        value = math.exp(weightedSum) * 1.0
    except OverflowError as exp:
        return 1
    return round((value) / (1.0 + value), 5)

def trainLR(data, vocab, I, E, L):
    wts = initializeWeights(vocab)
    for i in range(0, I):
        errorSummation = {}
        for classType in data:
            for item in data[classType]:
                features = getFeatures(item)
                classError = classValues[classType] - getClassProbability(features, wts)
                if classError != 0:
                    for feature in features.iterkeys():
                        if(errorSummation.has_key(feature)):
                            errorSummation[feature] += (features[feature] * classError)
                        else:
                            errorSummation[feature] = (features[feature] * classError)
        for wt in wts.iterkeys():
            if wt in errorSummation:
                wts[wt] = wts[wt] + (E * errorSummation[wt]) - (E * L * wts[wt])
    return wts

def testLR(wts):
    accuracy = {1: 0.0, 0: 0.0}
    for filename in os.listdir(os.getcwd() + '/test/' + 'ham'):
        features = getFileFeatures(os.getcwd() + '/test/' + 'ham', filename)
        classWeightedSum = getWeightedSum(features, wts)
        if(classWeightedSum >= 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    for filename in os.listdir(os.getcwd() + '/test/' + 'spam'):
        features = getFileFeatures(os.getcwd() + '/test/' + 'spam', filename)
        classWeightedSum = getWeightedSum(features, wts)
        if(classWeightedSum < 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    return (accuracy[1] * 100) / sum(accuracy.values())

k = 1000
e = 0.025
l = 0.1
i = 500

def main(args):

    k = int(args[1])
    e = float(args[2])
    l = float(args[3])
    i = int(args[4])

    stopWords = getStopWordsList('stop_words_list.txt')

    data = getData()

    vocab = buildVocab(data, [])
    restrictedVocab = buildVocab(data, stopWords)
    reducedVocab = featureSelection(data, vocab)

    prior, condProb = trainBN(data, vocab)
    print("Naive Bayes Accuracy with Stop Words : " + str(testBN(vocab, prior, condProb)))
    prior, condProb = trainBN(data, restrictedVocab)
    print("Naive Bayes Accuracy without Stop Words : " + str(testBN(restrictedVocab, prior, condProb)))
    prior, condProb = trainBN(data, reducedVocab)
    print("Naive Bayes Accuracy with Feature Selection : " + str(testBN(reducedVocab, prior, condProb)))

    wts = trainLR(data, vocab, i, e, l)
    print("Logistic Regression Accuracy with Stop Words : " + str(testLR(wts)))
    wts = trainLR(data, restrictedVocab, i, e, l)
    print("Logistic Regression Accuracy without Stop Words : " + str(testLR(wts)))
    wts = trainLR(data, reducedVocab, i, e, l)
    print("Logistic Regression Accuracy with Feature Selection : " + str(testLR(wts)))

if len(sys.argv) == 5:
	main(sys.argv)
else:
	print("Invalid number of arguments")
	print("Expected:")
	print("python spam_ham_filter.py <k> <e> <l> <i>")
	print("Where:")
	print("k: feature selection size.")
	print("e: eta value or the learning rate for Logistic Regression.")
	print("l: lambda value for Logistic Regression.")
	print("i: number of iterations for Logistic Regression.")


	
