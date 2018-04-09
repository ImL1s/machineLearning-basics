from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

# read csv file and get features&labels
allElectronicsData = open(r'./data.csv','rb')
reader = csv.reader(allElectronicsData)
headers = reader.next()
print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1,len(row) -1):
        rowDict[headers[i]] = row[i]

    featureList.append(rowDict)

print(featureList)
print(labelList)

# dummy X
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX: " + str(dummyX))
print("feature name: " + str(vec.get_feature_names()))

# dummy Y
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

# construct decision tree
# entropy equal ID3
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
print("clf: " + str(clf))

# export dot file
with open("allElectronicInformationGainOri.dot","w") as f:
    tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

# use graphviz cmd: dot -Tpdf allElectronicInformationGainOri.dot -o output.pdf

# test data
testRowX = dummyX[0, :]
print("testRowX: " + str(testRowX))
newRowX = testRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))
predictedY = clf.predict([newRowX])
print("predictedY: " +str(predictedY))
