# I have created
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def getKmers(sequence, size=6):
    return [sequence[x:x + size].lower() for x in range(len(sequence) - size + 1)]

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1



def result_class(request):
    if(request.method=='POST'):
        doc1 = request.FILES['document1']
        doc2 = request.FILES['document2']
    human = pd.read_table(doc1)
    chimp = pd.read_table(doc2)
    human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
    human = human.drop('sequence', axis=1)
    chimp['words'] = chimp.apply(lambda x: getKmers(x['sequence']), axis=1)
    chimp = chimp.drop('sequence', axis=1)
    human_texts = list(human['words'])
    for item in range(len(human_texts)):
        human_texts[item] = ' '.join(human_texts[item])
    y_h = human.iloc[:, 0].values
    chimp_texts = list(chimp['words'])
    for item in range(len(chimp_texts)):
        chimp_texts[item] = ' '.join(chimp_texts[item])
    y_c = chimp.iloc[:, 0].values

    cv = CountVectorizer(ngram_range=(4, 4))
    X = cv.fit_transform(human_texts)
    X_chimp = cv.transform(chimp_texts)
    # function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
    data1_shape=X.shape
    data2_shape=X_chimp.shape

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y_h,
                                                        test_size=0.2,
                                                        random_state=42)
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)


    conf_m1=pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted'))



    accuracy1, precision1, recall1, f11 = get_metrics(y_test, y_pred)
    #print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
    y_pred_chimp = classifier.predict(X_chimp)
    # performance on chimp genes
    #print("Confusion matrix\n")
    conf_m2=pd.crosstab(pd.Series(y_c, name='Actual'), pd.Series(y_pred_chimp, name='Predicted'))
    accuracy2, precision2, recall2, f12 = get_metrics(y_c, y_pred_chimp)
    #print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

    #print("see")

    # if f.mode=='r':
    #     contents=f.read()
    #     print(contents)
    # doc1=pd.read_table(doc1)
    params={'shape1':data1_shape,'shape2':data2_shape,'confu_matrix1':conf_m1,'confu_matrix2':conf_m2,'accuracy1':accuracy1,'precision1':precision1,'recall1':recall1,'f11':f11,'accuracy2':accuracy2,'precision2':precision2,'recall2':recall2,'f12':f12}
    return render(request, 'mysite/result_pred.html',params)

def result_newgene(request):
    if(request.method=='POST'):
        doc1 = request.FILES['document1']
        doc2 = request.FILES['document2']
    human = pd.read_table(doc1)
    chimp = pd.read_table(doc2)
    human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
    human = human.drop('sequence', axis=1)
    chimp['words'] = chimp.apply(lambda x: getKmers(x['sequence']), axis=1)
    chimp = chimp.drop('sequence', axis=1)
    human_texts = list(human['words'])
    for item in range(len(human_texts)):
        human_texts[item] = ' '.join(human_texts[item])
    y_h = human.iloc[:, 0].values
    chimp_texts = list(chimp['words'])
    for item in range(len(chimp_texts)):
        chimp_texts[item] = ' '.join(chimp_texts[item])
    y_c = chimp.iloc[:, 0].values

    cv = CountVectorizer(ngram_range=(4, 4))
    X = cv.fit_transform(human_texts)
    X_chimp = cv.transform(chimp_texts)
    # function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
    data1_shape=X.shape
    data2_shape=X_chimp.shape

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y_h,
                                                        test_size=0.2,
                                                        random_state=42)
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)


    conf_m1=pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted'))



    accuracy1, precision1, recall1, f11 = get_metrics(y_test, y_pred)
    #print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
    y_pred_chimp = classifier.predict(X_chimp)
    # performance on chimp genes
    #print("Confusion matrix\n")
    # conf_m2=pd.crosstab(pd.Series(y_c, name='Actual'), pd.Series(y_pred_chimp, name='Predicted'))
    # accuracy2, precision2, recall2, f12 = get_metrics(y_c, y_pred_chimp)
    #print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

    #print("see")
    res_gene=classifier.predict(X_chimp[0])
    # if f.mode=='r':
    #     contents=f.read()
    #     print(contents)
    # doc1=pd.read_table(doc1)
    params={'shape1':data1_shape,'shape2':data2_shape,'confu_matrix1':conf_m1,'accuracy1':accuracy1,'precision1':precision1,'recall1':recall1,'f11':f11,'ans':res_gene}
    return render(request, 'mysite/result_newgene_out.html',params)


