
############THIS IS USING SCIKIT LEARN###############

#import numpy as np
#from sklearn import cross_validation, neighbors, preprocessing
#import pandas as pd
#
#df = pd.read_csv("breast-cancer-wisconsin.data")
#df.replace('?',-99999,inplace=True)
#df.drop(['id'],1,inplace=True)
#
#X = np.array(df.drop(['class'],1))
#y = np.array(df['class'])
#
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#
#clf = neighbors.KNeighborsClassifier()
#
#clf.fit(X_train,y_train)
#
#accuracy = clf.score(X_test,y_test)
#print(accuracy)
#
#example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
#example_measures = example_measures.reshape(1,-1)
#prediction = clf.predict(example_measures)
#print(prediction)


#############THIS IS WITHOUT SCIKIT LEARN############

import pandas as pd
import random
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import style
import warnings
from collections import Counter
import numpy as np

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()

random.shuffle(full_data)
test_size=0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])

style.use('fivethirtyeight')

#dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
#new_features = [5,7]

def k_nearest_neighbors(data,predict,k=3):
	if len(data)>=k:
		warnings.warn("K is set to a value less than total voting groups...")
	
	distances=[]
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance, group])
	
	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result

correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		vote = k_nearest_neighbors(train_set,data,k=5)
		if group == vote:
			correct += 1
		total += 1
print("Accuracy=",correct/total)

#for i in dataset:
#	for ii in dataset[i]:
#		plt.scatter(ii[0],ii[1],s=30,color=i)

#plt.scatter(new_features[0],new_features[1],s=30)

#result = k_nearest_neighbors(dataset,new_features)
#plt.scatter(new_features[0],new_features[1],s=30,color=result)
#plt.show()
