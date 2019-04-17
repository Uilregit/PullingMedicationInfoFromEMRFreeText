import pandas as pd
import Levenshtein as lv

#guess = pd.read_csv("med2vec_results.csv",index_col = 0, header = None)

def checkIfLabelsInList (labels, target, threshold):
    if labels == [""]:
        return []
    else:
        labels = [i.lower() for i in labels]
    if target == [""]:
        return [False for i in labels]
    output = []
    for item in labels:
        if item in target:   #if exact match then return 0
            output += [True]
        elif item == "":
            output += []
        else:                   #if not an exact match
            levenDist = []
            wordsInLabels = len(item.split(" "))  #checks for multiword labels
            for i in range(len(target)-wordsInLabels+1):
                levenDist += [lv.distance(item, " ".join(target[i:i+wordsInLabels]))]   
            if (min(levenDist)/float(len(item)) <= threshold):
                output += [True]
            else:
                output += [False]
    return output

def get_metrics (modelLabels):
    truth = pd.read_csv("LabelsTest.csv", index_col = 0)
    guess = modelLabels

    truePositives = 0
    totalGuessList = 0
    totalTruthList = 0
    for i in guess.index:
        try:
            guessList = guess.loc[i].values[0].replace(","," ").split(" ")
        except AttributeError:
            guessList = [""]
        try:
            truthList = truth.loc[i].values[0].replace(", "," ").split(" ")
        except AttributeError:
            truthList = [""]
        truePositives += sum(checkIfLabelsInList(guessList, truthList, 0.2))
        totalGuessList += len(guessList)
        totalTruthList += len(truthList)

    falsePositives = totalGuessList - truePositives
    falseNegatives = totalTruthList - truePositives
    precision = truePositives/float(totalGuessList)
    recall = truePositives/float(totalTruthList)
    f1 = 0.5*precision*recall/(precision+recall)

    return precision, recall, f1