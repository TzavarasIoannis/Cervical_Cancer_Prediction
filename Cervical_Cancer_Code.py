import pandas as pd
import numpy as np
import matplotlib.pyplot    as plt
import seaborn as sns
import sklearn.metrics as m
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.neighbors import KNeighborsClassifier


#filepath
name = "C:\\Users\\tzav2\\Desktop\\ΠΜΣ\\Machine Learning\\risk_factors_cervical_cancer.csv"
#set nan values
missing_values_set = ["","--","?","na","NAN","nan" ]
#dataset attributes
attributes = ['Age', 'Number of sexual partners', 'First sexual intercourse (age)','Num of pregnancies'
,'Smokes','Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives', 'Hormonal Contraceptives (years)','IUD',
'IUD (years)','STDs','STDs (number)' 'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis',
              'STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV',
              'STDs: Number of diagnosis','STDs: Time since first diagnosis','STDs: Time since last diagnosis','Dx:Cancer''Dx:CIN','Dx:HPV','Dx',' Hinselmann','Schiller','Cytology', 'Biopsy']



#read file
with open(name) as file:
    dataset = pd.read_csv(file  ,na_values = missing_values_set )


print("mean of set  : ",dataset["Age"].mean())
print("Smokes \n ",dataset["Smokes"].value_counts())
print("Pregnance : \n", dataset["Num of pregnancies"].value_counts())
print("First sexual intercourse \n", dataset["First sexual intercourse"].value_counts())




print(dataset)
print(len(dataset))
######MISSING VALUES ########
print(dataset['Age'].value_counts())
print(  "NULL SUM:\n",dataset.isnull().sum().sum())









#
# plt.hist(dataset['Age'],bins=50)
# plt.xticks(range(0,80,5))
# plt.show()
print(dataset.info())
print(len(dataset.columns))
#missing values on each attribute
def missing_values(set):
    for att in set:
        print("number of NaN values in {} : {}".format(att, set[att].isnull().sum()))
missing_values(dataset)
dataset.dropna(axis=0 , thresh=18, inplace=True)  #drop rows that have over 36-18 Nans
#print(len(dataset)) #760
#replace values  with mean
def replace_values(set:pd.DataFrame):
    for att in set:
        mean = set[att].mean()
        mean = round(mean,2)        #limits the float
        #print(att , mean)
        set[att].fillna(mean,inplace=True)

replace_values(dataset)
#print(dataset.isnull().sum())




##### ATTRIBUTES ANALYSIS ########

corr = dataset.corr()                   #data correlation
print(dataset['STDs:AIDS'].unique())                      # unique values in column [0.]
print(dataset['STDs:cervical condylomatosis'].unique())   # unique values in column [0.]
# drop attr : 'STDs:cervical condylomatosis','STDs:AIDS', 'STDs: Time since first diagnosis ', 'STDs: Time since last diagnosis '
dataset.drop(columns=['STDs:cervical condylomatosis','STDs:AIDS', 'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']  ,inplace=True)
#print(len(dataset.iloc[0])) # number of attributes :32
#Hinselmann','Schiller','Cytology', 'Biopsy'


#print(dataset.columns)
for set in dataset:
    if len(dataset[set].unique()) == 1:
        print(set)

#dataset.to_csv(r'C:\Users\tzav2\Desktop\ΠΜΣ\Machine Learning\cancer_set1.csv')

######### correlation between #Hinselmann','Schiller','Cytology'     &       'Biopsy'

targets = dataset[['Hinselmann', 'Schiller', 'Citology','Biopsy']]

corr_targets = targets.corr()


mask1 = np.zeros_like(corr_targets)
mask1[np.triu_indices_from(mask1)] = True
sns.heatmap(corr_targets ,mask=mask1, cmap='PuOr',annot=True )
plt.title('Targets Correlations')
plt.show()


correlations = dataset.corr()

print(correlations)
print(correlations.sort_values(by='Biopsy'))
mask = np.zeros_like(correlations)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(correlations,mask=mask,xticklabels=correlations.columns , yticklabels=correlations.columns ,linewidths=0.5,cmap='Blues')  #annot=True,annot_kws={'size':7}
plt.title('Correlation Heatmap')
plt.show()



# print(dataset)
#
# print(len(dataset.columns))
# print(len(dataset))
#
#




dataset['Smokes (years)'] = round(dataset['Smokes (years)'],2)



cancer = dataset.to_csv(r'C:\Users\tzav2\Desktop\ΠΜΣ\Machine Learning\cancer.csv', index=False)



X= dataset[['Age', 'Number of sexual partners', 'First sexual intercourse',              ########### DATASET with 'Hinselmann', 'Schiller', 'Citology'
             'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
             'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
             'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
             'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
             'STDs:syphilis', 'STDs:pelvic inflammatory disease',
             'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:HIV',
             'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
             'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx','Hinselmann', 'Schiller', 'Citology']]


Y = dataset[['Biopsy']]


#split data



X_all_train , X_all_test, y_all_train , y_all_test = train_test_split(X , Y ,test_size=0.25)



from  sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
alpha = 0.015
clf = Lasso(alpha=alpha,normalize=True,fit_intercept=False)




def pretty_print_linear(coefs, names=None, sort=False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                      for coef, name in lst)

clf.fit(X_all_train, y_all_train)

print("LASSO OF unstandardized :")
#print(clf.coef_)
print("\n",pretty_print_linear(clf.coef_))
print("Training score of LASSO with alpha {} is {} \n".format(alpha, clf.score(X_all_train,y_all_train)))
print("Testing score of LASSO with alpha {} is {} \n".format(alpha, clf.score(X_all_test,y_all_test)))
print("clf != 0 : ",sum(clf.coef_!=0))
#

# #
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)
dataset =pd.DataFrame(dataset ,columns=['Age', 'Number of sexual partners', 'First sexual intercourse',
       'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
       'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
       'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
       'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
       'STDs:syphilis', 'STDs:pelvic inflammatory disease',
       'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:HIV',
       'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
       'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller',
       'Citology', 'Biopsy'])



X_scaled =dataset.drop(labels=['Biopsy'],axis=1)
Y_scaled = dataset['Biopsy']



X_all_train_scaled , X_all_test_scaled, y_all_train_scaled , y_all_test_scaled = train_test_split(X_scaled , Y_scaled ,test_size=0.25)
X_all_train_scaled = X_all_train_scaled.astype(int)
X_all_test_scaled= X_all_test_scaled.astype(int)
y_all_train_scaled= y_all_train_scaled.astype(int)
y_all_test_scaled= y_all_test_scaled.astype(int)

alpha=0.015
clf1 =Lasso(alpha=alpha,normalize=True,fit_intercept=False)

clf1.fit(X_all_train_scaled,y_all_train_scaled)
print("LASSO OF STARDIZED :")
print(clf1.coef_)
#print("\n",pretty_print_linear(clf1.coef_))
print("Training score of LASSO with alpha {} is {} \n".format(alpha, clf1.score(X_all_train_scaled,y_all_train_scaled)))
print("Testing score of LASSO with alpha {} is {} \n".format(alpha, clf1.score(X_all_test_scaled,y_all_test_scaled)))
print("clf != 0 : ",sum(clf1.coef_!=0))
print(X_all_train_scaled.columns)


features1 = [
"STDs: Number of diagnosis",
"STDs (number)",
"STDs",
"Dx:CIN",
"STDs:HIV",
"STDs:genital herpes",
"Dx:Cancer",
"Dx:HPV" ,
"Dx",
"Citology",
"Hinselmann",
"Schiller",


]



features  = [
'Hormonal Contraceptives (years)',
'STDs:condylomatosis',
'STDs:vulvo-perineal condylomatosis',
'STDs:genital herpes',
'STDs:HIV',
'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller',
       'Citology']





print("LENHGT : ",len(features))

X_new = X[features]
print(X_new)
print("LENGH X ",len(X_new))
print(X)

x_new_train ,x_new_test, y_new_train,  y_new_test = train_test_split(X_new,Y,test_size=0.25)
print(len(x_new_train))
print(len(y_new_train))
print(len(x_new_test))
print(len(y_new_test))



###################################################################### algorithms ############################################
### KNN
def KNN_algorithm(X_train ,X_test , y_train , y_test : pd.DataFrame):
    KNN = KNeighborsClassifier(n_neighbors=5 ,weights='distance',leaf_size=20)
    KNN.fit(X_train, y_train.values.ravel())

    # print(np.mean(cross_val_score(KNN ,X_all_train,y_all_train, cv=5)))

    predictions_from_knn = KNN.predict(X_test)
    print('confiusion matrix from knn with k=5 :\n {} \n '.format(m.confusion_matrix(y_test,predictions_from_knn)))
    print( 'Classification report from knn with k=5 \n : {} \n'.format(m.classification_report(y_test,predictions_from_knn)))
    print("knn k= accuracy :",m.accuracy_score(y_test,predictions_from_knn))
    print("precision : ", m.precision_score(y_test, predictions_from_knn))
    print("recall: ", m.recall_score(y_test, predictions_from_knn))
    print("f1 score: ", m.f1_score(y_test, predictions_from_knn,labels=np.unique(predictions_from_knn)))


   ########## ROC CURVE FOR KNN ################
    roc_acc_knn_re = m.roc_auc_score(y_test, KNN.predict_proba(X_test)[:,1])
    print("roc curve accuracy : ", roc_acc_knn_re)
    fpr ,tpr , thres = m.roc_curve(y_test,KNN.predict_proba(X_test)[:,1])
    figure= m.RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_acc_knn_re,estimator_name='KNN')
    figure.plot()
    plt.title("roc curve kNN")
    plt.show()




#SVM

def SVM_algorithm(X_train ,X_test , y_train , y_test : pd.DataFrame):

    from sklearn.svm import SVC
    svm_classifier = SVC(kernel='linear',probability=True )
    svm_classifier.fit(X_train,y_train.values.ravel())
    svm_predi  = svm_classifier.predict(X_test)


    print('confiusion matrix :\n {} \n '.format(m.confusion_matrix(y_test, svm_predi)))
    print('Classification report SVM  \n : {} \n'.format(m.classification_report(y_test, svm_predi)))
    print("SVMs accuracy :", m.accuracy_score(y_test, svm_predi))
    print("SVM precision : ", m.precision_score(y_test, svm_predi,pos_label=1))
    print("SVM recall: ", m.recall_score(y_test, svm_predi, average='binary'  ,pos_label=1))
    print("SVM f1 score: ", m.f1_score(y_test, svm_predi,labels=np.unique(svm_predi), pos_label=1))


######### ROC CURVE FOR SVM ################
    svm_roc = m.roc_auc_score(y_test,svm_classifier.predict_proba(X_test)[:,1])
    print("roc curve accuracy : ", svm_roc)
    fpr , tpr , thresh = m.roc_curve(y_test,svm_classifier.predict_proba(X_test)[:,1] , pos_label=1)
    figure_svm = m.RocCurveDisplay(fpr=fpr ,tpr=tpr , roc_auc=svm_roc , estimator_name="SVM")
    figure_svm.plot()
    plt.show()

#GAUSS


def GAUSSIAN_algorithm(X_train ,X_test , y_train , y_test : pd.DataFrame):

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train,y_train.values.ravel())
    gnb_predctions =  gnb.predict(X_test)

    print('confiusion matrix from gaussianNB :\n {} \n '.format(m.confusion_matrix(y_test, gnb_predctions)))
    print('Classification report GaussianNB  \n : {} \n'.format(m.classification_report(y_test, gnb_predctions)))
    print("GaussianNB accuracy :", m.accuracy_score(y_test, gnb_predctions))
    print("GaussianNB precision : ", m.precision_score(y_test, gnb_predctions))
    print("GaussianNB recall: ", m.recall_score(y_test, gnb_predctions))
    print("GaussianNB f1 score: ", m.f1_score(y_test, gnb_predctions,labels=np.unique(gnb_predctions)))

    fpr_gnb , tpr_gnb , _ = m.roc_curve(y_test,gnb.predict_proba(X_test)[:,1])
    gnb_roc_acc = m.roc_auc_score(y_test, gnb.predict_proba(X_test)[:,1])
    print("roc curve accuracy : ", gnb_roc_acc)
    bayes_fig = m.RocCurveDisplay(fpr=fpr_gnb , tpr=tpr_gnb , estimator_name="BAYES" , roc_auc=gnb_roc_acc)
    bayes_fig.plot()
    plt.show()


## NNs

def NN_algorithm(X_train ,X_test , y_train , y_test : pd.DataFrame):

    from sklearn.neural_network import MLPClassifier

# # #‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
# # #‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
# # # hidden layer size : the ith element represents the number of neurons in the ith hidden layer.
    nn = MLPClassifier(hidden_layer_sizes=(30,30,30) ,activation="relu",solver='lbfgs',alpha=1e-5,random_state=1,max_iter=1000)
    nn.fit(X_train,y_train.values.ravel())
    nn_predictions = nn.predict(X_test)

    print('confiusion matrix from NNs :\n {} \n '.format(m.confusion_matrix(y_test, nn_predictions)))
    print('NNs report   \n : {} \n'.format(m.classification_report(y_test, nn_predictions)))
    print("NNs accuracy :", m.accuracy_score(y_test, nn_predictions))
    print("NNs precision : ", m.precision_score(y_test, nn_predictions))
    print("NNs recall: ", m.recall_score(y_test, nn_predictions))
    print("NNs f1 score: ", m.f1_score(y_test, nn_predictions, labels=np.unique(nn_predictions)))


#####################  ROC CURVE ACCURACY  #############################################

    fpr_nn , tpr_nn , t_ = m.roc_curve(y_test,nn.predict_proba(X_test)[:,1])
    nn_roc_acc = m.roc_auc_score(y_test, nn.predict_proba(X_test)[:,1])
    print("roc curve accuracy : ", nn_roc_acc)
    nn_fig = m.RocCurveDisplay(fpr=fpr_nn , tpr=tpr_nn , estimator_name="NNs" , roc_auc=nn_roc_acc)
    nn_fig.plot()
    plt.show()







# #
KNN_algorithm(x_new_train ,x_new_test, y_new_train,  y_new_test )
# SVM_algorithm(x_new_train ,x_new_test, y_new_train,  y_new_test )
# GAUSSIAN_algorithm(x_new_train ,x_new_test, y_new_train,  y_new_test )
# NN_algorithm(x_new_train ,x_new_test, y_new_train,  y_new_test )
# # # # #
#


