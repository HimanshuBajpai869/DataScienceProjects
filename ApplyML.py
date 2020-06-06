from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,roc_auc_score, accuracy_score
from xgboost import XGBClassifier

tree_classifier = DecisionTreeClassifier()
logistic_classifier = LogisticRegression()
svc_classifier = SVC(kernel= 'rbf')
nb_classifier = GaussianNB()
knn_classifier = KNeighborsClassifier(n_neighbors= 2 , metric = 'minkowski', p = 2)
xgboost_clf = XGBClassifier()

def FitAndEvaluateMLModel(x_train, y_train, x_test, y_test):
    # Fit the decision tree classifier in the data set
    tree_classifier.fit(x_train,y_train)

    # Make predictions
    tree_pred = tree_classifier.predict(x_test)
    PrintAccuracy(classifier_name= 'Decision Tree', y_pred= tree_pred, y_test= y_test)
    
    print(classification_report(y_test, tree_pred))
    
    logistic_classifier.fit(x_train,y_train)

    # Make predictions
    logistic_pred = logistic_classifier.predict(x_test)
    PrintAccuracy(classifier_name= 'Logistic Regression', y_pred= logistic_pred, y_test= y_test)
    
    print(classification_report(y_test, logistic_pred))
    
    # Fit SVM into the dataset
    svc_classifier.fit(x_train, y_train)

    # Make predictions
    svc_pred = svc_classifier.predict(x_test)
    PrintAccuracy(classifier_name= 'Support Vector Machine', y_pred= svc_pred, y_test= y_test)
    
    print(classification_report(y_test, svc_pred))
    
    # Fit Naive Bayes to the dataset

    nb_classifier.fit(x_train, y_train)

    # Make predictions
    nb_pred = nb_classifier.predict(x_test)
    PrintAccuracy(classifier_name= 'Naive Bayes', y_pred= nb_pred, y_test= y_test)
    
    print(classification_report(y_test, nb_pred))
    
    # Fit KNN to the dataset
    knn_classifier.fit(x_train, y_train)

    # Make predictions
    knn_pred = knn_classifier.predict(x_test)
    PrintAccuracy(classifier_name= 'KNN', y_pred= knn_pred, y_test= y_test)
    
    print(classification_report(y_test, knn_pred))
    
    # N- Fold Cross Validations for the different model used

    # 1. Decision Tree Classifier
    print(f'Average accuracy of Decision Tree Classifier after applying 10- Fold CV is {ApplyCrossValidation(tree_classifier, x_train, y_train)}')
    # 91.07203630175837

    # 2. Logistic Regression
    print(f'Average accuracy of Logistic Regression after applying 10- Fold CV is {ApplyCrossValidation(logistic_classifier, x_train, y_train)}')
    # 95.25808281338628

    # 3. SVM
    print(f'Average accuracy of SVM after applying 10- Fold CV is {ApplyCrossValidation(svc_classifier, x_train, y_train)}')
    # 62.91548496880317

    # 4. Naive Bayes
    print(f'Average accuracy of Naive Bayes after applying 10- Fold CV is {ApplyCrossValidation(nb_classifier, x_train, y_train)}')
    # 93.85138967668748

    # 5. KNN
    print(f'Average accuracy of KNN after applying 10- Fold CV is {ApplyCrossValidation(knn_classifier, x_train, y_train)}')
    

def PrintAccuracy(classifier_name, y_test, y_pred):
    print(f'Accuracy using {classifier_name} is :- ', (accuracy_score(y_test,y_pred)*100))

def ApplyCrossValidation(regressor, x_train, y_train):
    accuracies = cross_val_score(estimator = regressor, X = x_train, y = y_train, cv = 10)
    return accuracies.mean()*100

def get_classification_model(algo_name):
    if algo_name == 'DT':
        return tree_classifier
    elif algo_name == 'LR':
        return logistic_classifier
    elif algo_name == 'SVM':
        return svc_classifier
    elif algo_name == 'NB':
        return nb_classifier
    elif algo_name == 'KNN':
        return knn_classifier
    elif algo_name == "XGB":
        return xgboost_clf
    
def apply_classification_model(model_name, x_train,x_test,y_train,y_test):
    print(f'Fitting and evaluating the {model_name} model to the dataset.')
    model = get_classification_model(model_name)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    print(f'ROC-AUC Score for test set - {roc_auc_score(y_test, y_pred)}')
    print(f'Accuracy for test set - {accuracy_score(y_test, y_pred)*100}')
    print(f'Accuracy for training set - {accuracy_score(y_train, y_pred_train)*100}')         
          