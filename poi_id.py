import sys
import pickle
import numpy as np
import pprint
from collections import defaultdict
sys.path.append("../tools/")

#importing a couple of functions from feature_format.py
from feature_format import featureFormat, targetFeatureSplit
#importing a couple of functions from tester.py
from tester import dump_classifier_and_data
from tester import test_classifier

#all 21 features in the dataset inside org_features_list
orig_features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                    'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                    'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                    'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                    'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#counting how many POI and NON-POI in the dataset
#also counting how many NULL in each feature
num_poi = 0
num_non_poi = 0
cnt_null = defaultdict(int)
for n, f in data_dict.iteritems():
    if f['poi']:
        num_poi += 1
    else:
        num_non_poi += 1
    for k,v in f.iteritems():
        if v == 'NaN':
            cnt_null[k] += 1
            
            
    
#using the name of the chairman of Enron Corporation
num_features = len(data_dict['LAY KENNETH L'])

print("Total count of people in the data set: {}".format(len(data_dict)))
print("Total count of persons of interest (POI) in the data set : {}".format(num_poi))
print("Total count of non persons of interest (NON POI) in the data set : {}".format(num_non_poi))
print("Each person has {} features and are listed below :\n".format(num_features))
#listing the names of all the features
for k in data_dict['LAY KENNETH L'].keys():
    print k
print "\nTotal number of missing values for each feature:"
#listing the total number of missing values for each feature
for k in cnt_null.keys():
    print(k,cnt_null[k])

### Remove outliers
#we will check the data to see if there is any outlier

#using the two features 'salary' and 'bonus' to check for any outlier
feat = ["salary", "bonus"]
dt = featureFormat(data_dict, feat)
#plotting the two features
import matplotlib.pyplot as plt
for pt in dt:
    sal = pt[0]
    bon = pt[1]
    plt.scatter(sal, bon)

plt.xlabel("salary")
plt.ylabel("bonus")
print "\nChecking for outliers: "
plt.show()

#checking the data manually, the outlier is 'TOTAL' and it will be removed
data_dict.pop('TOTAL', None)
#removing the datapoint called 'THE TRAVEL AGENCY IN THE PARK' since it does not relate to the Enron employees
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)
#plotting the two features with outliers removed
dt = featureFormat(data_dict, feat)
for pt in dt:
    sal = pt[0]
    bon = pt[1]
    plt.scatter(sal, bon)

plt.xlabel("salary")
plt.ylabel("bonus")
print "\n Scatterplot after removing the 'TOTAL' outlier: "
plt.show()

#finding the top 4 outliers
outliers = []
for k in data_dict:
    v = data_dict[k]['salary']
    if v == 'NaN':
        continue
    outliers.append((k, int(v)))

outliers_high = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
print "The 4 highest outliers:\n"
print outliers_high
#I will not remove these outliers since they belong to our POI

### Store to orig_dataset for easy export below.
orig_dataset = data_dict

### Extract features and labels from dataset for local testing
#we included all of the features in the dataset inside orig_features_list
dt = featureFormat(orig_dataset, orig_features_list, sort_keys = True)
labels, feat = targetFeatureSplit(dt)

#scaling features
from sklearn import preprocessing
scaled_minmax = preprocessing.MinMaxScaler()
scaled_feat = scaled_minmax.fit_transform(feat)
#splitting the data for testing and training
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(scaled_feat, labels, test_size=0.1, random_state=42)

#Trying a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from time import time

#creating a function that obtains a type of classifier and predicts after fitting
#this function also outputs the accuracy, precision, and recall scores of the classifier
def algo(clf):
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print "Accuracy :",accuracy_score(pred, labels_test)
    print "Precision Score :",precision_score(pred, labels_test)
    print "Recall Score :",recall_score(pred, labels_test)

print "\n==============Performance metrics of different algorithms before adding the new features==============="
print "\nNaive Bayes Results :"
algo(GaussianNB())

print "\nDecision Tree Results :"
algo(DecisionTreeClassifier())

print "\nSupport Vector Machine Results :"
algo(SVC())

print "\nRandom Forest Results :"
algo(RandomForestClassifier())

###  Create new feature(s)

#creating two new features 'fraction_from_this_person_to_poi' and 'fraction_from_poi_to_this_person
for n, f in data_dict.items():
    if f['from_messages'] == 'NaN' or f['from_this_person_to_poi'] == 'NaN':
        f['fraction_from_this_person_to_poi'] = 0.0
    else:
        f['fraction_from_this_person_to_poi'] = \
                                    f['from_this_person_to_poi'] / float(f['from_messages'])

    if f['to_messages'] == 'NaN' or f['from_poi_to_this_person'] == 'NaN':
        f['fraction_from_poi_to_this_person'] = 0.0
    else:
        f['fraction_from_poi_to_this_person'] = \
                                    f['from_poi_to_this_person'] / float(f['to_messages'])
            
#following the same method as before to measure the performance of algorithms after adding the new features.
my_dataset = data_dict

#adding my two newly created features into the orig_features_list
orig_features_list.extend(['fraction_from_poi_to_this_person','fraction_from_this_person_to_poi'])


dt = featureFormat(my_dataset, orig_features_list, sort_keys = True)
labels, feat = targetFeatureSplit(dt)

#scaling features
scaled_minmax = preprocessing.MinMaxScaler()
scaled_feat = scaled_minmax.fit_transform(feat)

features_train, features_test, labels_train, labels_test = train_test_split(scaled_feat, labels, test_size=0.2, random_state=42)

#lets see if there's any change to the performance after adding the new features

print "\n==============Performance metrics of different algorithms after adding the new features==============="
print "\nNaive Bayes Results :"
algo(GaussianNB())

print "\nDecision Tree Results :"
algo(DecisionTreeClassifier())

print "\nSupport Vector Machine Results :"
algo(SVC())

print "\nRandom Forest Results :"
algo(RandomForestClassifier())

#a function called 'score_func' that counts the total number of True Negative, False Negative,
#True Positive, and False Positive and using those to calculate the precision, recall, and f1 scores
def score_func(y_true,y_predict):
    true_neg = 0
    false_neg = 0
    true_pos = 0
    false_pos = 0

    for predict, right in zip(y_predict, y_true):
        if predict == 0 and right == 0:
            true_neg += 1
        elif predict == 0 and right == 1:
            false_neg += 1
        elif predict == 1 and right == 0:
            false_pos += 1
        else:
            true_pos += 1
    if true_pos == 0:
        return (0,0,0)
    else:
        precision = 1.0*true_pos/(true_pos+false_pos)
        recall = 1.0*true_pos/(true_pos+false_neg)
        fscore = 2.0 * true_pos/(2*true_pos + false_pos+false_neg)
        return (precision,recall,fscore)

#a function called univariateFeatureSelection that calculates each feature's precision, recall, and
#fscore by using the 'score_func' function
def univariateFeatureSelection(f_list, my_dataset):
    result = []
    for f in f_list:
        #replacing 'NaN' with 0
        for n in my_dataset:
            dt_pt = my_dataset[n]
            if not dt_pt[f]:
                dt_pt[f] = 0
            elif dt_pt[f] == 'NaN':
                dt_pt[f] = 0

        dt = featureFormat(my_dataset, ['poi',f], sort_keys = True, remove_all_zeroes = False)
        labels, feat = targetFeatureSplit(dt)
        feat = [abs(x) for x in feat]
        from sklearn.model_selection import StratifiedShuffleSplit
        cv = StratifiedShuffleSplit(n_splits=1000,test_size=0.1,random_state = 42)
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for train_idx, test_idx in cv.split(feat, labels):
            for s in train_idx:
                features_train.append( feat[s] )
                labels_train.append( labels[s] )
            for h in test_idx:
                features_test.append( feat[h] )
                labels_test.append( labels[h] )
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        score = score_func(labels_test,predictions)
        result.append((f,score[0],score[1],score[2]))
    result = sorted(result, reverse=True, key=lambda x: x[3])
    return result 

#listing all the features with their precision, recall, and f scores.
#Univariate Feature Selection
univariate_result = univariateFeatureSelection(orig_features_list,my_dataset)
print '\n### univariate feature selection result'
for j in univariate_result:
    print j
    
#these are the features I selected for my features_list
features_list = ['poi','total_stock_value','exercised_stock_options','bonus','deferred_income','long_term_incentive',
                    'restricted_stock','salary','total_payments','other', 'shared_receipt_with_poi']
                         
dt = featureFormat(my_dataset, features_list, sort_keys = True)
labels, feat = targetFeatureSplit(dt)

#scaling features
scaled_minmax = preprocessing.MinMaxScaler()
scaled_feat = scaled_minmax.fit_transform(feat)

features_train, features_test, labels_train, labels_test = train_test_split(scaled_feat, labels, test_size=0.1, random_state=42)

print "\n==============Performance metrics of different algorithms after selecting my own features==============="
print "\nNaive Bayes Results :"
algo(GaussianNB())

print "\nDecision Tree Results :"
algo(DecisionTreeClassifier())

print "\nSupport Vector Machine Results :"
algo(SVC())

print "\nRandom Forest Results :"
algo(RandomForestClassifier())

###  Tune your classifier to achieve better than .3 precision and recall 
features_train, features_test, labels_train, labels_test = \
                                    train_test_split(feat, labels, test_size=0.3, random_state=42)
    
print "\n==============Decision Tree's Performance Metrics==============="
decision_tree = DecisionTreeClassifier()

dt_par = [{'min_samples_split': [2,3,4], 'criterion': ['gini', 'entropy']}]

#using GridSearchCV to pick the best possible parameters for Decision Tree
dt_grid = GridSearchCV(estimator = decision_tree,\
                       param_grid = dt_par,\
                       cv = StratifiedKFold(n_splits = 6, shuffle = True),\
                       n_jobs = -1,\
                       scoring = 'f1')

#measuring the Decision Tree's training time
start_fit = time()
dt_grid.fit(features_train, labels_train)
end_fit = time()
print("Training time : {}".format(round(end_fit - start_fit, 3)))

#measuring the Decision Tree's predicting time 
start_pred = time()
dt_pred = dt_grid.predict(features_test)
end_pred = time()
print("Predicting time : {}".format(round(end_pred - start_pred, 3)))

#outputting Decision Tree's performance metrics
dt_acc = accuracy_score(dt_pred, labels_test)
print('Decision Tree accuracy : {}'.format(dt_acc))
print "F1 score :",f1_score(dt_pred, labels_test)
print "Precision score :",precision_score(dt_pred, labels_test)
print "Recall score :",recall_score(dt_pred, labels_test)
print(dt_grid.best_estimator_)

print "\n==============Naive Bayes's Performance Metrics==============="

naive = GaussianNB()
nb_pipe = Pipeline([('scaler', MinMaxScaler()),('selection', SelectKBest()),('pca', PCA()),('naive_bayes', naive)])

nb_par = [{ 'selection__k': [8,9,10], 'pca__n_components': [6,7,8] }]

#using GridSearchCV to pick the best possible parameters for Naive Bayes
nb_grid = GridSearchCV(estimator = nb_pipe,\
                        param_grid = nb_par,\
                        n_jobs = -1,\
                        cv = StratifiedKFold(n_splits = 6, shuffle = True),\
                        scoring = 'f1')

#measuring the Naive Bayes's training time
start_fit = time()
nb_grid.fit(features_train, labels_train)
end_fit = time()
print("Training time : {}".format(end_fit - start_fit))

#measuring the Naive Bayes's predicting time 
start_pred = time()
nb_pred = nb_grid.predict(features_test)
end_pred = time()
print("Predicting time : {}".format(end_pred - start_pred))

#outputting Naive Bayes's performance metrics
nb_acc = accuracy_score(nb_pred, labels_test)
print('Naive Bayes accuracy : {}'.format(nb_acc))
print "F1 score :",f1_score(nb_pred, labels_test)
print "Precision score :",precision_score(nb_pred, labels_test)
print "Recall score :",recall_score(nb_pred, labels_test)
print(nb_grid.best_estimator_)

print "\n==============Support Vector Classification's Performance Metrics==============="

svc_pipe = Pipeline([('scaler',MinMaxScaler()), ('svc', SVC())])

svc_par = { 'svc__kernel': ['linear','rbf'],
                   'svc__C': [0.1,1,10,100,1000],
                   'svc__gamma': [1e-3,1e-4,1e-1,1,10] }
                 
#using GridSearchCV to pick the best possible parameters for Support Vector Classification
svc_grid = GridSearchCV(estimator = svc_pipe,\
                        param_grid = svc_par,\
                        cv = StratifiedKFold(n_splits = 6, shuffle = True),\
                        n_jobs = -1,\
                        scoring = 'f1')

#measuring the Support Vector Classification's training time
start_fit = time()
svc_grid.fit(features_train, labels_train)
end_fit = time()
print("Training time : {}".format(end_fit - start_fit))

#measuring the Support Vector Classification's predicting time
start_pred = time()
svc_pred = svc_grid.predict(features_test)
end_pred = time()
print("Predicting time : {}".format(end_pred - start_pred))

#outputting Support Vector Classification's performance metrics
svc_acc = accuracy_score(svc_pred, labels_test)
print('SVC accuracy score : {}'.format(svc_acc))
print "F1 score :",f1_score(svc_pred, labels_test)
print "Precision score :",precision_score(svc_pred, labels_test)
print "Recall score :",recall_score(svc_pred, labels_test)
svc_best_est = svc_grid.best_estimator_
print(svc_best_est)

#choosing Naive Bayes's as my final algorithm
print "\n==============Final Algorithm: Naive Bayes==============="
test_classifier(nb_grid.best_estimator_, my_dataset, features_list)

#checking to see if addition of the new feature on my selected feature_list will improve the performance

experiment_features_list = ['poi','total_stock_value','exercised_stock_options','bonus','deferred_income','long_term_incentive',
    'restricted_stock','salary','total_payments','other', 'shared_receipt_with_poi','fraction_from_this_person_to_poi']

print "\n=================Effect of the new feature on the final algorithm: Naive Bayes================="
test_classifier(nb_grid.best_estimator_, my_dataset, experiment_features_list)

###Task 6: Dump your classifier, dataset, and features_list so anyone can check your results
dump_classifier_and_data(nb_grid.best_estimator_, my_dataset, features_list)