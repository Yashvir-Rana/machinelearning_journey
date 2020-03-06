from sklearn import tree
features = [[130,1],[140,1],[150,0],[170,0]] # 0 for bumpy, 1 for smooth
labels = [0, 0, 1, 1]   # 0 for orange , 1 for apple

clf = tree.DecisionTreeClassifier()   # define the model
clf = clf.fit(features, labels)    # fit data in model

print (clf.predict([[146, 1]]))   # predict the label