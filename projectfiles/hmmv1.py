#!/usr/bin/env python
# coding: utf-8

# Yap Hui Xuan, Rachel : A0185579H Lee Jia Yao, Edward : A0183660A Lew Kee Siong Lionel : A0185418X Low Xin Hui: A0188265R

# # Question 2

# In[48]:
pip install pandas
pip install numpy

import pandas as pd
import numpy as np
import json
import random
from itertools import groupby
import math


# In[49]:


ddir = "."


# In[50]:


def mle(token,tag,df):
    num_words = len(df[0].unique()) #number of unique words
    sigma = 0.1
    num = len(df.loc[(df[0]==token) & (df[1]== tag)])  #how many times word is associated with tag
    denom = len(df.loc[df[1]==tag]) #how many times the tag appears
    return (num + sigma)/(denom+ sigma*(num_words+1))
    #return num/denom


# In[51]:


def txtToDf(input_file):
    file = open(input_file, encoding="utf8") #added encoding 
    df = file.read()
    df = df.splitlines()
    df = pd.Series(df)
    df = df.str.split(expand=True)
    df = pd.DataFrame(df)
    return df


# In[52]:


test = txtToDf(f'{ddir}/twitter_train.txt')


# In[53]:


for i in range (len(test)-1):
    token = test.loc[i, 0]
    tag = test.loc[i,1]
    test.loc[i, 'MLE'] = mle(token, tag,test)


# In[54]:


test.dropna(inplace = True)


# In[55]:


np.savetxt(f'{ddir}/naive_output_probs.txt',test.values,fmt='%s',delimiter='\t', encoding='utf-8') #writing naive_putput_probs.txt


# In[56]:


def smoothing(df):
    tag = ""
    prob = 0
    sigma = 0.1
    num_words = len(df)
    df = df.dropna()
    lis = df[1].unique() #unique tags
    for i in lis:
        temp = sigma/(len(df.loc[df[1]==i]) + sigma*(num_words+1)) #for unseen words
        if temp > prob:#get max probablity for unseen token 
            tag = i
            prob = temp
            temp = 0
    return (tag,prob)


# In[57]:


def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    train = txtToDf(in_output_probs_filename) #naive output prob
    train = train.drop_duplicates()
    train = train.sort_values(by=[0])
    idx = train.groupby([0])[2].transform(max) == train[2] #
    #group by words, get max MLE, form another col, MLE = max MLE for the same words
    
    trainClean = train[idx] #only pick those with true values
    
    test = txtToDf(in_test_filename) #'twitter_dev_no_tag.txt', only tokens
    out = test.merge(trainClean, how='left', sort = False) #merge train and test data, on token, NA = words not found in train 
    
    out[0].fillna(" ",inplace=True) #words in trainclean but not test = ""
    unseen = smoothing(out) #gives the tag with the largest probability
    out[1].fillna(unseen[0], inplace = True)
    out[2].fillna(unseen[1],inplace = True)
    out = out[out[0] != " "] # only select those words in test

    np.savetxt(out_prediction_filename ,out[1].values, fmt ='%s',delimiter="\t", encoding='utf-8')


# In[58]:


# inp = f'{ddir}/naive_output_probs.txt'
# tes = f'{ddir}/twitter_dev_no_tag.txt'


# In[59]:


# naive_predict(inp, tes, f'{ddir}/naive_predictions.txt')


# # Question 3: Improved Naive Approach

# In[60]:


def helper(token,tag,df):
    num = (len(df.loc[df[1]==tag])) #count(tags)
    denom =(len(df.loc[(df[0]==token)])) #count(words)
    return num/denom


# In[61]:


def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    improved = txtToDf(in_output_probs_filename) #naive output prob
    improved = improved.drop_duplicates()
    improved.reset_index(drop = True, inplace=True) #reset index for looping 
    improved = improved.sort_values(by=[0])
    
    train = txtToDf(in_train_filename) #use twitter_train to get P(y=j)/P(x=w), place in col 2
    for i in range (len(train)-1):
        token = train.loc[i, 0]
        tag = train.loc[i,1]
        if (token == None or tag == None):
            train.loc[i, 2] = None
        else:
            train.loc[i, 2] = helper(token, tag,train)
    train.dropna(inplace = True)
    train.reset_index(drop = True, inplace=True) #reset index for looping 
    train = train.drop_duplicates()
    train.reset_index(drop = True, inplace=True) #reset index for looping 
    train = train.sort_values(by=[0])
    
    #multiply P(y=j)/P(x=w) by MLE from q2 
    for i in range(len(improved)-1):
        improved.loc[i,2] = float(improved.loc[i,2])*float(train.loc[i,2])

    #group by words, get max MLE, form another col, MLE = max MLE for the same words
    idx = improved.groupby([0])[2].transform(max) == improved[2]
    
    trainClean = improved[idx] #only pick those with true values
    
    test = txtToDf(in_test_filename) #'twitter_dev_no_tag.txt', only tokens
    
    out = test.merge(trainClean, how='left', sort = False) #merge train and test data, on token, NA = words not found in train 
    out[0].fillna(" ",inplace=True) #words in trainclean but not test = ""
    unseen = smoothing(out) #gives the tag with the largest probability
    out[1].fillna(unseen[0], inplace = True)
    out[2].fillna(unseen[1],inplace = True)
    out = out[out[0] != " "] # only select those words in test

    np.savetxt(out_prediction_filename ,out[1].values, fmt ='%s',delimiter="\t", encoding='utf-8')


# In[62]:


# inp = f'{ddir}/naive_output_probs.txt'
# intrain = f'{ddir}/twitter_train.txt'
# tes = f'{ddir}/twitter_dev_no_tag.txt'


# In[63]:


# naive_predict2(inp, intrain, tes, f'{ddir}/naive_predictions2.txt')


# # Question 4 : Viterbi Algorithm 

# In[64]:


def tokenTags(token,emission_prob):
    temp = [] #tags of tokens
    for k in emission_prob:
        for k2 in emission_prob[k]:
            if k2 == token:
                temp.append(k)
    return temp


# In[65]:


def get_transition_prob(in_train_file_name):
    trainRaw = txtToDf(in_train_file_name)
    train = trainRaw.fillna('START') #all empty rows indicate the end of a tweet, replace end of tweet with start
    temp = pd.Series(['START','START']) 
    temp = pd.DataFrame([temp])
    temp #add START symbol for first tweet
    train = pd.concat([temp,train], ignore_index=True)
    
    tagsList = train[1].tolist()
    tagsDict = {}
    for i in tagsList: #get unique tags
        if i not in tagsDict:
            tagsDict[i] = {}
        
    for i in tagsDict: #count the the number of transitions from state i to state j
        for j in range(0,len(tagsList)-1):
            if i == tagsList[j]:
                if tagsList[j+1] in tagsDict[i]:
                    tagsDict[i][tagsList[j+1]] += 1
                else:
                    tagsDict[i][tagsList[j+1]] = 1
    
    for i in tagsDict: #transition prob from X-START indicates X-STOP
        if i != 'START':
            try:
                temp = tagsDict[i]['START']
                tagsDict[i]['STOP'] = temp
                del tagsDict[i]['START']
            except:
                pass
    
    transition_prob = {}
    for k in tagsDict:
        transition_prob[k] = {} #initialize unique tags in transition_prob
    
    sigma = 0.0001
    num_words = len(trainRaw.dropna())
    
    
    for k in transition_prob:
        for j in tagsDict[k].keys():
            total = sum(tagsDict[k].values())
            transition_prob[k][j] = (tagsDict[k][j] + sigma)/(total + sigma*(num_words+1))
    return transition_prob


# In[66]:


transition_prob=get_transition_prob(f'{ddir}/twitter_train.txt')
with open(f'{ddir}/trans_probs.txt', 'w') as file:
    file.write(json.dumps(transition_prob)) # use `json.loads` to do the reverse


# In[67]:


def get_emission_prob(in_train_filename):
    train = txtToDf(in_train_filename)
    train = train.rename(columns={0:'word',1:'tags'})
    trainCount = train.groupby(train.columns.tolist()).size().reset_index().        rename(columns={0:'records'})
    trainCountList= trainCount.values.tolist()
    wordTagDict = {}
    for i in trainCountList:
        if i[1] not in wordTagDict:
            wordTagDict[i[1]] = {} #initialize all unique tags as keys
    for i in wordTagDict:
        for j in trainCountList:
            if i == j[1]:
                wordTagDict[i][j[0]] = j[2]
    emission_prob = {}
    sigma = 0.0001
    num_words = len(train.dropna())
    for k in wordTagDict:
        emission_prob[k] = {}

        total = sum(wordTagDict[k].values())
        for j in wordTagDict[k].keys():
            emission_prob[k][j] = (wordTagDict[k][j]+sigma)/(total+sigma*(num_words)+1)
        #all unseen tokens collapse into 'allUnseenTokens'
        emission_prob[k]['allUnseenTokens'] = sigma/(total+sigma*(num_words)+1)
    return emission_prob


# In[68]:


emission_prob = get_emission_prob(f'{ddir}/twitter_train.txt')
with open(f'{ddir}/output_probs.txt', 'w') as file:
    file.write(json.dumps(emission_prob)) # use `json.loads` to do the reverse


# In[69]:


from itertools import groupby
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    #read test file
    file = open(in_test_filename, encoding = "utf8")
    corpus = file.read()
    corpus = corpus.splitlines()
    #load emission probs
    with open(in_output_probs_filename, encoding = "utf8") as json_file:
        emission_prob = json.load(json_file)
    #load transition probs
    with open(in_trans_probs_filename, encoding = "utf8") as json_file:
        transition_prob = json.load(json_file)
    
    #splitting input into sublist of tweets
    res = [list(sub) for ele, sub in groupby(corpus, key = bool) if ele]
    
    #add start and stop sybmbols for each tweet
    for i in range(0,len(res)):
        res[i].insert(0,'START')
        res[i].append('STOP')
    
    predicted_tags = []

    for i in range(0,len(res)):
        tweets = res[i] #sentences
        pis = {}
        for j in range(0,len(tweets)):
            observedOutput = tweets[j] #tokens
            #append predicted tags from pis to predicted_tags list at the end of every sentence
            if observedOutput =='STOP': 
                pisList = list(pis.values())
                for i in range(0,len(pisList)):
                    if i>0:
                        for j in list(pisList[i].keys()):
                            predicted_tags.append(j)
                pis = {}
                break
            
            #list of all possible tags for current token, if unseen token, empty list returned
            possibleTags = tokenTags(observedOutput,emission_prob)
            if not possibleTags:
                observedOutput = 'allUnseenTokens'
            possibleTags = tokenTags(observedOutput,emission_prob)
            
            if 'STOP' in possibleTags:
                possibleTags.remove('STOP')
            
            pis[j] = {}
            temp = []
            if j == 1:
                for k in possibleTags:
                    if k in transition_prob['START']:
                        #get all Pis from state 0 (start) to state 1
                        temp.append(transition_prob['START'][k] * emission_prob[k][observedOutput])
                    else:
                        temp.append(0)

                #getting the state with the highest probability in list temp
                indexOfMax = temp.index(max(temp))
                tag = possibleTags[indexOfMax]
                #add the most likely state in pis
                pis[j][tag] = max(temp)
                continue

            if j>1:
                previous_state = list(pis[j-1].keys())[0]
                temp =[]

                for k in possibleTags:
                    if k in transition_prob[previous_state]:
                        temp.append(transition_prob[previous_state][k] * emission_prob[k][observedOutput])
                    else:
                        temp.append(0)

                indexOfMax = temp.index(max(temp))
                tag = possibleTags[indexOfMax]
                pis[j][tag] = max(temp)
                continue
    with open(out_predictions_filename, "w") as fhandle:
        for tags in predicted_tags:
            fhandle.write(f'{tags}\n')


# In[70]:


# in_tags = f'{ddir}/twitter_tags.txt'
# in_trans = f'{ddir}/trans_probs.txt'
# in_emission = f'{ddir}/output_probs.txt'
# in_emi = f'{ddir}/naive_output_probs.txt'
# in_test = f'{ddir}/twitter_dev_no_tag.txt'
# in_ans = f'{ddir}/twitter_dev_ans.txt'
# in_output = f'{ddir}/viterbi_predictions.txt'


# In[71]:


#viterbi_predict(in_tags, in_trans, in_emission, in_test,in_output)


# # Question 5

# In[72]:


transition_prob=get_transition_prob(f'{ddir}/twitter_train.txt')
with open(f'{ddir}/trans_probs2.txt', 'w') as file:
    file.write(json.dumps(transition_prob)) # use `json.loads` to do the reverse


# In[73]:


emission_prob = get_emission_prob(f'{ddir}/twitter_train.txt')
with open(f'{ddir}/output_probs2.txt', 'w') as file:
    file.write(json.dumps(emission_prob)) # use `json.loads` to do the reverse


# In[74]:


def txtToSuffixDf2(input_file):
    file = open(input_file, encoding="utf8")
    df = file.read()
    df = df.splitlines()
    df = pd.Series(df)
    df = df.str.split(expand=True)
    df = pd.DataFrame(df)
    for i in range(len(df)):
        token = df.loc[i,0]
        if token:
            if '@USER' not in token:
                df.loc[i,0] = token[-2:]
    return df


# In[75]:


def get_emission_prob_suffix2(in_train_filename):
    train = txtToSuffixDf2(in_train_filename)
    train = train.rename(columns={0:'suffix',1:'tags'})
    #.groupby.size(): returns number of rows for each set of words and tags as a Series
    #traincount returns group of  (words, tags) and its count 
    trainCount = train.groupby(train.columns.tolist()).size().reset_index().rename(columns={0:'records'})
    trainCountList= trainCount.values.tolist() #make each row into a list and store in list
    
    #print(trainCountList)
    wordTagDict = {} # keys = tag, value = {}
    for i in trainCountList:
        if i[1] not in wordTagDict:
            wordTagDict[i[1]] = {} #initialize all unique tags as keys
    for i in wordTagDict:
        for j in trainCountList:
            if i == j[1]:
                wordTagDict[i][j[0]] = j[2] #{tag:{word:count}}
    emission_prob = {}
    sigma = 0.0001
    num_words = len(train.dropna())
    for k in wordTagDict:
        emission_prob[k] = {} # {tags:{}}

        total = sum(wordTagDict[k].values()) #number of tags (no duplicates)
        for j in wordTagDict[k].keys():
            emission_prob[k][j] = (wordTagDict[k][j]+sigma)/(total+sigma*(num_words)+1) #{tag:{word:probability}}
    return emission_prob


# In[76]:


emission_prob_suffix2 = get_emission_prob_suffix2(f'{ddir}/twitter_train.txt')


# In[77]:


def txtToSuffixDf3(input_file):
    file = open(input_file, encoding="utf8")
    df = file.read()
    df = df.splitlines()
    df = pd.Series(df)
    df = df.str.split(expand=True)
    df = pd.DataFrame(df)
    for i in range(len(df)):
        token = df.loc[i,0]
        if token:
            if '@USER' not in token:
                df.loc[i,0] = token[-3:]
    return df


# In[78]:


def get_emission_prob_suffix3(in_train_filename):
    train = txtToSuffixDf3(in_train_filename)
    train = train.rename(columns={0:'suffix',1:'tags'})
    #.groupby.size(): returns number of rows for each set of words and tags as a Series
    #traincount returns group of  (words, tags) and its count 
    trainCount = train.groupby(train.columns.tolist()).size().reset_index().rename(columns={0:'records'})
    trainCountList= trainCount.values.tolist() #make each row into a list and store in list
    
    #print(trainCountList)
    wordTagDict = {} # keys = tag, value = {}
    for i in trainCountList:
        if i[1] not in wordTagDict:
            wordTagDict[i[1]] = {} #initialize all unique tags as keys
    for i in wordTagDict:
        for j in trainCountList:
            if i == j[1]:
                wordTagDict[i][j[0]] = j[2] #{tag:{word:count}}
    emission_prob = {}
    sigma = 0.0001
    num_words = len(train.dropna())
    for k in wordTagDict:
        emission_prob[k] = {} # {tags:{}}

        total = sum(wordTagDict[k].values()) #number of tags (no duplicates)
        for j in wordTagDict[k].keys():
            emission_prob[k][j] = (wordTagDict[k][j]+sigma)/(total+sigma*(num_words)+1) #{tag:{word:probability}}
    return emission_prob


# In[79]:


emission_prob_suffix3 = get_emission_prob_suffix3(f'{ddir}/twitter_train.txt')


# In[80]:


def txtToSuffixDf4(input_file):
    file = open(input_file, encoding="utf8")
    df = file.read()
    df = df.splitlines()
    df = pd.Series(df)
    df = df.str.split(expand=True)
    df = pd.DataFrame(df)
    for i in range(len(df)):
        token = df.loc[i,0]
        if token:
            if '@USER' not in token:
                df.loc[i,0] = token[-4:]
    return df


# In[81]:


def get_emission_prob_suffix4(in_train_filename):
    train = txtToSuffixDf4(in_train_filename)
    train = train.rename(columns={0:'suffix',1:'tags'})
    #.groupby.size(): returns number of rows for each set of words and tags as a Series
    #traincount returns group of  (words, tags) and its count 
    trainCount = train.groupby(train.columns.tolist()).size().reset_index().rename(columns={0:'records'})
    trainCountList= trainCount.values.tolist() #make each row into a list and store in list
    
    #print(trainCountList)
    wordTagDict = {} # keys = tag, value = {}
    for i in trainCountList:
        if i[1] not in wordTagDict:
            wordTagDict[i[1]] = {} #initialize all unique tags as keys
    for i in wordTagDict:
        for j in trainCountList:
            if i == j[1]:
                wordTagDict[i][j[0]] = j[2] #{tag:{word:count}}
    emission_prob = {}
    sigma = 0.0001
    num_words = len(train.dropna())
    for k in wordTagDict:
        emission_prob[k] = {} # {tags:{}}

        total = sum(wordTagDict[k].values()) #number of tags (no duplicates)
        for j in wordTagDict[k].keys():
            emission_prob[k][j] = (wordTagDict[k][j]+sigma)/(total+sigma*(num_words)+1) #{tag:{word:probability}}
    return emission_prob


# In[82]:


emission_prob_suffix4 = get_emission_prob_suffix4(f'{ddir}/twitter_train.txt')


# In[127]:


import json
from itertools import groupby
def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    #read test file
    file = open(in_test_filename,encoding = "utf8")
    corpus = file.read()
    corpus = corpus.splitlines()
    #load emission probs
    with open(in_output_probs_filename,encoding = "utf8") as json_file:
        emission_prob = json.load(json_file)
    #load transition probs
    with open(in_trans_probs_filename,encoding = "utf8") as json_file:
        transition_prob = json.load(json_file)
    
    #splitting input into sublist of tweets
    res = [list(sub) for ele, sub in groupby(corpus, key = bool) if ele]
    
    #add start and stop sybmbols for each tweet
    for i in range(0,len(res)):
        res[i].insert(0,'START')
        res[i].append('STOP')
        for j in range(0,len(res[i])):
            if '@USER' in res[i][j]:
                res[i][j] = '@USER'
    
    predicted_tags = []
    #count = 0
    for i in range(0,len(res)):
        tweets = res[i] #sentences
        pis = {}
        for j in range(0,len(tweets)):
            observedOutput = tweets[j] #tokens
            
            #append predicted tags from pis to predicted_tags list at the end of every sentence
            if observedOutput =='STOP': 
                pisList = list(pis.values())
                for i in range(0,len(pisList)):
                    if i>0:
                        for j in list(pisList[i].keys()):
                            predicted_tags.append(j)
                pis = {}
                break

            #list of all possible tags for current token, if unseen token, empty list returned
            possibleTags = tokenTags(observedOutput, emission_prob) 
            if 'STOP' in possibleTags:
                possibleTags.remove('STOP')
            
            pis[j] = {}
            temp = []
            if j == 1:
                if "@USER" in observedOutput:
                    tag = '@'
                    pis[j][tag] = 1
                    continue
                if "http" in observedOutput:
                    tag = 'U'
                    pis[j][tag] = 1
                    continue
                try:
                    if float(observedOutput):
                        tag = '$'
                        pis[j][tag] = 1
                        continue
                except:
                    pass
                        
                try:
                    for k in possibleTags:
                        #get all Pis from state 0 (start) to state 1
                        temp.append(transition_prob['START'][k] * emission_prob[k][observedOutput])

                    #getting the state with the highest probability in list temp
                    indexOfMax = temp.index(max(temp))
                    tag = possibleTags[indexOfMax]
                    #add the most likely state in pis
                    pis[j][tag] = max(temp)

                except:
                    try:
                        #append emission probability with different suffixes and get the one with max probabilty
                        possibleTags_suffix4 = tokenTags(observedOutput[-4:], emission_prob_suffix4)
                        temp4 = []
                        for k in possibleTags_suffix4:
                            temp4.append(transition_prob['START'][k] * emission_prob_suffix4[k][observedOutput[-4:]])
                    
                        possibleTags_suffix3 = tokenTags(observedOutput[-3:], emission_prob_suffix3)
                        temp3 = []
                        for k in possibleTags_suffix3:
                            temp3.append(transition_prob['START'][k] * emission_prob_suffix3[k][observedOutput[-3:]])
                        
                        possibleTags_suffix2 = tokenTags(observedOutput[-2:], emission_prob_suffix2)
                        temp2 = []
                        for k in possibleTags_suffix2:
                            temp2.append(transition_prob['START'][k] * emission_prob_suffix2[k][observedOutput[-2:]])
                        possibleTags_dict = {'possibleTags_suffix2': possibleTags_suffix2, 'possibleTags_suffix3': possibleTags_suffix3, 'possibleTags_suffix4': possibleTags_suffix4}
                        tempArr_dict = {'temp2': temp2, 'temp3': temp3, 'temp4':temp4}
                        
                        #getting the state with the highest probability in list temp
                        #temp = list(map(lambda x: max(x),[temp2,temp3,temp4]))
                        temp = [max(i) if i else 0 for i in [temp2,temp3,temp4]]
                        if max(temp) == 0:
                            raise Exception
                        indexOfMaxTemp = temp.index(max(temp))
                        indexOfMax = tempArr_dict['temp'+ str(indexOfMaxTemp +2)].index(temp[indexOfMaxTemp])
                        tag = possibleTags_dict['possibleTags_suffix' + str(indexOfMaxTemp +2)][indexOfMax]
                        #add the most likely state in pis
                        pis[j][tag] = max(temp)
                        
                    except:
                        if 'STOP' in transition_prob['START'].keys():
                            new = transition_prob['START'].copy()
                            del new['STOP']
                            tag = list(new.keys())[list(new.values()).index(max(new.values()))]
                            pis[j][tag] = max(new.values())
                        else:
                            tag = list(transition_prob['START'].keys())[list(transition_prob['START'].values()).index(max(transition_prob['START'].values()))]
                            pis[j][tag] = max(transition_prob['START'].values())


            if j>1:
                previous_state = list(pis[j-1].keys())[0]
                temp =[]
                if "@USER" in observedOutput:
                    tag = '@'
                    pis[j][tag] = 1
                    continue
                if "http" in observedOutput:
                    tag = 'U'
                    pis[j][tag] = 1
                    continue
                try:
                    if float(observedOutput):
                        tag = '$'
                        pis[j][tag] = 1
                        continue
                except:
                    pass
                try:
                    for k in possibleTags:
                        #get all Pis from state 0 (start) to state 1
                        temp.append(transition_prob[previous_state][k] * emission_prob[k][observedOutput])

                    #getting the state with the highest probability in list temp
                    indexOfMax = temp.index(max(temp))
                    tag = possibleTags[indexOfMax]
                    #add the most likely state in pis
                    pis[j][tag] = max(temp)

                except:
                    try:
                        #append emission probability with different suffixes and get the one with max probabilty
                        possibleTags_suffix4 = tokenTags(observedOutput[-4:], emission_prob_suffix4)
                        temp4 = []
                        for k in possibleTags_suffix4:
                            temp4.append(transition_prob['START'][k] * emission_prob_suffix4[k][observedOutput[-4:]])
                        
                        possibleTags_suffix3 = tokenTags(observedOutput[-3:], emission_prob_suffix3)
                        temp3 = []
                        for k in possibleTags_suffix3:
                            temp3.append(transition_prob['START'][k] * emission_prob_suffix3[k][observedOutput[-3:]])
                        
                        possibleTags_suffix2 = tokenTags(observedOutput[-2:], emission_prob_suffix2)
                        temp2 = []
                        for k in possibleTags_suffix2:
                            temp2.append(transition_prob['START'][k] * emission_prob_suffix2[k][observedOutput[-2:]])
                        
                        possibleTags_dict = {'possibleTags_suffix2': possibleTags_suffix2, 'possibleTags_suffix3': possibleTags_suffix3, 'possibleTags_suffix4': possibleTags_suffix4}
                        tempArr_dict = {'temp2': temp2, 'temp3': temp3, 'temp4':temp4}
                        
                        #getting the state with the highest probability in list temp
                        temp = [max(i) if i else 0 for i in [temp2,temp3,temp4]]
                        if max(temp) == 0:
                            raise Exception
                        indexOfMaxTemp = temp.index(max(temp))
                        indexOfMax = tempArr_dict['temp'+ str(indexOfMaxTemp +2)].index(temp[indexOfMaxTemp])
                        tag = possibleTags_dict['possibleTags_suffix' + str(indexOfMaxTemp +2)][indexOfMax]
                        #add the most likely state in pis
                        pis[j][tag] = max(temp)

                    except:
                        if 'STOP' in transition_prob[previous_state].keys():
                            new = transition_prob[previous_state].copy()
                            del new['STOP']
                            tag = list(new.keys())[list(new.values()).index(max(new.values()))]
                            pis[j][tag] =max(new.values())
                        else:
                            tag = list(transition_prob[previous_state].keys())[list(transition_prob[previous_state].values()).index(max(transition_prob[previous_state].values()))]
                            pis[j][tag] =max(transition_prob[previous_state].values())

    with open(out_predictions_filename, "w") as fhandle:
        for tags in predicted_tags:
            fhandle.write(f'{tags}\n')


# In[84]:


# in_tags = f'{ddir}/twitter_tags.txt'
# in_trans = f'{ddir}/trans_probs2.txt'
# in_emission = f'{ddir}/output_probs2.txt'
# in_emi = f'{ddir}/naive_output_probs.txt'
# in_test = f'{ddir}/twitter_dev_no_tag.txt'
# in_ans = f'{ddir}/twitter_dev_ans.txt'
# in_output = f'{ddir}/viterbi_predictions2.txt'


# In[85]:


#viterbi_predict2(in_tags, in_trans, in_emission, in_test, in_output)


# # Question 6

# In[86]:


#randomly initialise transition matrix
def init_trans(in_tag_filename,seed):
    random.seed(seed)
    tags = open(in_tag_filename)
    tags = tags.read().splitlines()
    trans_dict = {} #randomise transition probabilities from each state to all other states (incl itself)
    trans_dict["<START>"]={}
    trans_dict["<START>"]["<STOP>"] = 0
    trans_dict['<STOP>']={}
    for i in tags:

        trans_dict["<START>"][i] = random.random()
        trans_dict[i] = {}
        trans_dict[i]["<START>"]=0
        for j in tags:
            trans_dict[i][j] = random.random()
        trans_dict['<STOP>'][i]=0
        trans_dict[i]["<STOP>"] = random.random() #randomise transition from any state to stop state since not given
    
    #normalise probs, scaling
    for i in trans_dict:
        try:
            total = sum(trans_dict[i].values())
            factor = 1/total
            for j in trans_dict[i]:
                trans_dict[i][j] = trans_dict[i][j]*factor
            #print(sum(trans_dict[i].values()))
        except:
            continue
    trans_dict["<START>"]["<START>"]=0
    trans_dict["<STOP>"]["<STOP>"]=0
    trans_dict["<STOP>"]["<START>"] = 0
        
    return trans_dict


# In[87]:


def init_output(in_train_filename, in_tag_filename, seed):
    random.seed(seed)
    output_dict = {}
    file=open(in_train_filename, encoding="utf8")
    corpus = file.read()
    corpus = corpus.splitlines()

    #splitting input into sublist of tweets
    res = [list(sub) for ele, sub in groupby(corpus, key = bool) if ele]

    #add start and stop sybmbols for each tweet
    for i in range(0,len(res)):
        res[i].insert(0,'START_WORD')
        res[i].append('STOP_WORD')
    #print(res)
    #tokens = res[0]
    tags = open(in_tag_filename)
    tags = tags.read().splitlines()
    #tags.append("<START>")
    #tags.append("<STOP>")

    for i in tags:
        output_dict[i] = {}
        for j in res:
            for k in j:
                if i == "<START>" and k == "START_WORD":
                    output_dict[i][k]=1
                elif i == "<STOP>" and k == "STOP_WORD":
                    output_dict[i][k]= 1
                elif i=="<START>":
                    output_dict[i][k]=0
                elif i =="<STOP>":
                    output_dict[i][k] = 0
                elif k:
                    if k == "START_WORD":
                        output_dict[i][k]=0
                    elif k == "STOP_WORD":
                        output_dict[i][k]=0
                    else:
                        output_dict[i][k] = random.random()
    #normalise
    for i in output_dict:
        try:
            total = sum(output_dict[i].values())
            for j in output_dict[i]:
                output_dict[i][j] = output_dict[i][j]/total
        except:
            continue
    return output_dict


# In[88]:


#Alpha matrix for one tweet
def forward_alpha(trans_dict, output_dict, tweet, tags):#compute alpha for one tweet, add stop after each tweet in final algo
    alpha_matrix = {}
    alpha_matrix[0] ={}
    
    #initialise 1st col for emission prob of one token in all states 
    
    for tag in tags:
        alpha_matrix[0][tag] = trans_dict["<START>"][tag]*output_dict[tag][tweet[1]]
    denom = sum(alpha_matrix[0].values())
    
    for i in range(1,len(tweet)):
        
        if i == len(tweet)-1:
            break
        
        prev = i - 1
        alpha_matrix[i] = {}
        
        if i==len(tweet)-2:
            count = 0
            for tag in tags:
                count += alpha_matrix[prev][tag]*trans_dict[tag]["<STOP>"]
            alpha_matrix[i]["<STOP>"]=count

        else:
            denom = 0
            for tag_curr in tags:
                count = 0
                for tag_old in tags:
                    count += alpha_matrix[prev][tag_old] * trans_dict[tag_old][tag_curr]*output_dict[tag_curr][tweet[i+1]]
                alpha_matrix[i][tag_curr] = count
            denom = sum(alpha_matrix[i].values())
            #for t in tags:
            #    alpha_matrix[i][t] = alpha_matrix[i][t]/denom

    return alpha_matrix


# In[89]:


#beta matrix for one tweet
def backward_beta(trans_dict, output_dict, tweet, tags):#compute beta for one tweet
    #tweet = tweet[:-1] #remove STOP appended 
    beta_matrix = {}
    beta_matrix[len(tweet) - 2] ={}
    
    #initialise 1st col for emission prob of one token in all states 
    for tag in tags:
        beta_matrix[len(tweet) - 2][tag] = trans_dict[tag]["<STOP>"]

    for i in range(len(tweet)-3, -1, -1):

        prev = i + 1
        beta_matrix[i] = {}
        if i==0:
            count = 0
            for old_tag in tags:
                count += trans_dict["<START>"][old_tag]*output_dict[old_tag][tweet[prev]]*beta_matrix[prev][old_tag]
            beta_matrix[0]["<START>"] = count

        else:
            for tag_curr in tags:
                count = 0
                for old_tag in tags:
                    count += trans_dict[tag_curr][old_tag] * output_dict[old_tag][tweet[prev]]                    * beta_matrix[prev][old_tag]
                beta_matrix[i][tag_curr] = count

    return beta_matrix


# In[90]:


def gamma(alpha_matrix, beta_matrix, tweet, tags): #alpha matrix from forward function, beta_matrix from backward function
    gamma_dict = {}
    for i in range(len(tweet)-2):
        gamma_dict[i] = {}
        for tag in tags:
            gamma_dict[i][tag] = alpha_matrix[i][tag]*beta_matrix[i+1][tag]/            alpha_matrix[len(tweet)-2]['<STOP>']
        #denom = sum(gamma_dict[i].values())
        #for tag in gamma_dict[i]:
        #    gamma_dict[i][tag] = gamma_dict[i][tag]/denom

    return gamma_dict


# In[91]:


def xi(alpha_matrix, beta_matrix, init_trans, init_output, tweet, tags): #alpha matrix from forward function, beta_matrix from backward function
    xi_dict = {}

    for k in range(len(tweet)):
        if k == len(tweet)-1:
            break
        
        xi_dict[k] = {} # {index:{tag_i:{tag_j: prob}}}
        
        if k ==0:
            xi_dict[k]["<START>"]={}
    
        for i in tags:
            if k == 0:                    #alpha0 * trans[start][j] * output[j][tweet+1]* beta1
                xi_dict[k]["<START>"][i]= 1*init_output[i][tweet[k+1]]*init_trans["<START>"][i]                *beta_matrix[k+1][i]/ alpha_matrix[len(tweet)-2]['<STOP>']
                #print(xi_dict[0])
            elif k == len(tweet)-2:
                xi_dict[k][i]={} #alphaN * trans[i][stop] * output[stop][stop] *betaN
                xi_dict[k][i]["<STOP>"] = alpha_matrix[k-1][i]*init_trans[i]["<STOP>"]                *1/ alpha_matrix[len(tweet)-2]['<STOP>']
            else:
                xi_dict[k][i] = {}
                for j in tags:
                    xi_dict[k][i][j] = alpha_matrix[k-1][i]*init_trans[i][j]*init_output[j][tweet[k+1]]                    *beta_matrix[k+1][j] / alpha_matrix[len(tweet)-2]['<STOP>']


#             for t1 in xi_dict[k]:
#                 denom = sum(xi_dict[k][t1].values())
#                 for j in xi_dict[k][t1]:
#                     xi_dict[k][t1][j] = xi_dict[k][t1][j]/denom
    
    return xi_dict 


# In[92]:


#Finding new transition prob // for 1 tweet only
def update_trans(xi_matrix_list,tags):
    
    new_trans = {}
    new_trans["<START>"]={}
    new_trans["<STOP>"]= {}
    for i in tags:
        new_trans[i]={}
        
    for a in range(len(xi_matrix_list)):
        mat1 = xi_matrix_list[a]
        for i in range(len(mat1.keys())):
            tags = list(mat1[i].keys())
            for j in tags:
                tags2 = list(mat1[i][j].keys())
                for k in tags2:
                    if k in new_trans[j].keys():
                        curr = new_trans[j][k]
                        curr += mat1[i][j][k]
                        new_trans[j][k] = curr
                    else:
                        new_trans[j][k] = mat1[i][j][k]

    for q in new_trans:
        denom = sum(new_trans[q].values())
#         if denom ==0:
#             print(q)
#             print(new_trans[q])
        for l in new_trans[q]:
            try:
                new_trans[q][l] = new_trans[q][l]/denom
            except:
                continue
    return new_trans


# In[93]:


def update_output(gamma_matrix_list,tweets,tags):
    new_output = {}
    for i in tags:
        new_output["<START>"] = {}
        new_output[i] = {}
        new_output["<STOP>"] = {}
    
    new_output["<START>"]["START_WORD"]=1
    new_output["<STOP>"]["STOP_WORD"]=1
    
    #print(new_output)
    for i in range(len(gamma_matrix_list)):
        gamma_matrix = gamma_matrix_list[i]
        tweet = tweets[i]
        for tag in tags:
            for j in range(len(tweet)):
                if tweet[j] == "START_WORD":
                    continue
                elif tweet[j] == "STOP_WORD":
                    continue
                else:
                    if tweet[j] in new_output[tag]:
                        new_output[tag][tweet[j]] += gamma_matrix[j-1][tag]
                    else:
                        new_output[tag][tweet[j]] = gamma_matrix[j-1][tag]
    #print(new_output)
    for q in new_output:
        denom = sum(new_output[q].values())
        for l in new_output[q]:
            try:
                new_output[q][l] = new_output[q][l]/denom
            except:
                continue


    return new_output
    


# In[94]:


def forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh):
    random.seed(seed)
    file = in_train_filename
    #read test file
    file = open(file, encoding = "utf8")
    corpus = file.read().splitlines()

    tags = open(in_tag_filename,encoding = "utf8" )
    tags = tags.read().splitlines()
    
    #splitting input into sublist of tweets
    tweets = [list(sub) for ele, sub in groupby(corpus, key = bool) if ele]

    #add start and stop sybmbols for each tweet
    for i in range(0,len(tweets)):
        tweets[i].insert(0,'START_WORD')
        tweets[i].append('STOP_WORD')
    
    init_aMatrix = init_trans(in_tag_filename,seed)
    init_bMatrix = init_output(in_train_filename,in_tag_filename,seed)
    with open(f'{ddir}/trans_probs3.txt', 'w') as file:
        file.write(json.dumps(init_aMatrix))
    with open(f'{ddir}/output_probs3.txt', 'w') as file:
        file.write(json.dumps(init_bMatrix))
    
    #log_likelihoodArr = []

    for j in range(max_iter):

        #print(j)
        gamma_list = []
        xi_list = []
        log_likelihood = 0
        
        for tweet in tweets:
            alpha = forward_alpha(init_aMatrix,init_bMatrix,tweet,tags)
            #print(alpha[len(tweet)-2]["<STOP>"])
           
            for i in alpha:
                if "<STOP>" in alpha[i]:
                    log_likelihood += math.log(alpha[i]["<STOP>"])
            
            beta = backward_beta(init_aMatrix,init_bMatrix,tweet,tags)
            gammaTerm = gamma(alpha,beta,tweet,tags)
            xiTerm = xi(alpha, beta, init_aMatrix,init_bMatrix,tweet,tags)
            gamma_list.append(gammaTerm)
            xi_list.append(xiTerm)

        init_aMatrix = update_trans(xi_list,tags)
        init_bMatrix = update_output(gamma_list,tweets,tags)

        #print(log_likelihood)
        #log_likelihoodArr.append(log_likelihood)
        #print(init_aMatrix)
        
    with open(out_trans_filename, 'w') as file:
        file.write(json.dumps(init_aMatrix))
    with open(out_output_filename, 'w') as file:
        file.write(json.dumps(init_bMatrix))
    
    return


# In[95]:


# in_train_filename =  f'{ddir}/twitter_train_no_tag.txt'
# in_tag_filename =  f'{ddir}/twitter_tags.txt'
# out_trans_filename = f'{ddir}/trans_probs4.txt'
# out_output_filename = f'{ddir}/output_probs4.txt'


# In[96]:


#forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,10,8,1e-4)


# ### VITERBI

# In[97]:


def txtToSuffixDf2(input_file):
    file = open(input_file, encoding="utf8")
    df = file.read()
    df = df.splitlines()
    df = pd.Series(df)
    df = df.str.split(expand=True)
    df = pd.DataFrame(df)
    for i in range(len(df)):
        token = df.loc[i,0]
        if token:
            if '@USER' not in token:
                df.loc[i,0] = token[-2:]
    return df


# In[98]:


def get_emission_prob_suffix2_fb(in_train_filename, in_tag_filename, emission_prob_file):
    train = txtToSuffixDf2(in_train_filename)
    train.dropna(inplace = True)
    train = list(pd.unique(train[0]))
    tags = list(txtToDf(in_tag_filename)[0])
    with open(emission_prob_file,encoding = "utf8") as json_file:
        emission_prob = json.load(json_file)
    emission_prob_suffix = {} # {suffix:prob}
    
    #load randomised emission probability from above, add up prob for all tokens with same suffix
    for i in tags:
        for j in train: #go through suffixes of length = 2
            emission_prob_suffix[j] = 0
            for k in emission_prob[i]:
                if "@USER" not in k and k[-2:] == j:
                    emission_prob_suffix[j] += emission_prob[i][k]
    
    emission_prob_new = {}
    for i in tags:
        emission_prob_new[i] = {}
        for j in train:
            if "@USER" in j:
                emission_prob_new[i][j] = emission_prob[i][j]
            else:
                emission_prob_new[i][j] = emission_prob_suffix[j]

    for i in emission_prob_new:
        for j in emission_prob_new[i]:
            emission_prob_new[i][j] /= sum(list(emission_prob_new[i].values()))
    return emission_prob_new


# In[99]:


def txtToSuffixDf3(input_file):
    file = open(input_file, encoding="utf8")
    df = file.read()
    df = df.splitlines()
    df = pd.Series(df)
    df = df.str.split(expand=True)
    df = pd.DataFrame(df)
    for i in range(len(df)):
        token = df.loc[i,0]
        if token:
            if '@USER' not in token:
                df.loc[i,0] = token[-3:]
    return df


# In[100]:


def get_emission_prob_suffix3_fb(in_train_filename, in_tag_filename, emission_prob_file):

    train = txtToSuffixDf3(in_train_filename)
    train.dropna(inplace = True)
    train = list(pd.unique(train[0]))
    tags = list(txtToDf(in_tag_filename)[0])
    with open(emission_prob_file,encoding = "utf8") as json_file:
        emission_prob = json.load(json_file)
    emission_prob_suffix = {} # {suffix:prob}
    
    #load randomised emission probability from above, add up prob for all tokens with same suffix
    for i in tags:
        for j in train: #go through suffixes of length = 3
            emission_prob_suffix[j] = 0
            for k in emission_prob[i]:
                if "@USER" not in k and k[-3:] == j:
                    emission_prob_suffix[j] += emission_prob[i][k]
    
    emission_prob_new = {}
    for i in tags:
        emission_prob_new[i] = {}
        for j in train:
            if "@USER" in j:
                emission_prob_new[i][j] = emission_prob[i][j]
            else:
                emission_prob_new[i][j] = emission_prob_suffix[j]

    for i in emission_prob_new:
        for j in emission_prob_new[i]:
            emission_prob_new[i][j] /= sum(list(emission_prob_new[i].values()))
    return emission_prob_new


# In[101]:


def txtToSuffixDf4(input_file):
    file = open(input_file, encoding="utf8")
    df = file.read()
    df = df.splitlines()
    df = pd.Series(df)
    df = df.str.split(expand=True)
    df = pd.DataFrame(df)
    for i in range(len(df)):
        token = df.loc[i,0]
        if token:
            if '@USER' not in token:
                df.loc[i,0] = token[-4:]
    return df


# In[102]:


def get_emission_prob_suffix4_fb(in_train_filename, in_tag_filename, emission_prob_file):
    
    train = txtToSuffixDf4(in_train_filename)
    train.dropna(inplace = True)
    train = list(pd.unique(train[0]))
    tags = list(txtToDf(in_tag_filename)[0])
    with open(emission_prob_file,encoding = "utf8") as json_file:
        emission_prob = json.load(json_file)
    emission_prob_suffix = {} # {suffix:prob}
    
    #load randomised emission probability from above, add up prob for all tokens with same suffix
    for i in tags:
        for j in train: #go through suffixes of length = 4
            emission_prob_suffix[j] = 0
            for k in emission_prob[i]:
                if "@USER" not in k and k[-4:] == j:
                    emission_prob_suffix[j] += emission_prob[i][k]
    
    emission_prob_new = {}
    for i in tags:
        emission_prob_new[i] = {}
        for j in train:
            if "@USER" in j:
                emission_prob_new[i][j] = emission_prob[i][j]
            else:
                emission_prob_new[i][j] = emission_prob_suffix[j]

    for i in emission_prob_new:
        for j in emission_prob_new[i]:
            emission_prob_new[i][j] /= sum(list(emission_prob_new[i].values()))
    return emission_prob_new


# In[103]:


def tokenTags(token,emission_prob):
    temp = [] #tags of tokens
    for k in emission_prob:
        for k2 in emission_prob[k]:
            if k2 == token:
                temp.append(k)
    return temp


# In[126]:


def viterbi_predict2_fb(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    #read test file
    file = open(in_test_filename,encoding = "utf8")
    corpus = file.read()
    corpus = corpus.splitlines()
    #load emission probs
    with open(in_output_probs_filename,encoding = "utf8") as json_file:
        emission_prob = json.load(json_file)
    #load transition probs
    with open(in_trans_probs_filename,encoding = "utf8") as json_file:
        transition_prob = json.load(json_file)
    
    emission_prob_suffix2 = get_emission_prob_suffix2_fb(f'{ddir}/twitter_train_no_tag.txt',in_tags_filename,in_output_probs_filename)
    emission_prob_suffix3 = get_emission_prob_suffix3_fb(f'{ddir}/twitter_train_no_tag.txt',in_tags_filename,in_output_probs_filename)
    emission_prob_suffix4 = get_emission_prob_suffix4_fb(f'{ddir}/twitter_train_no_tag.txt',in_tags_filename,in_output_probs_filename)
    
    #splitting input into sublist of tweets
    res = [list(sub) for ele, sub in groupby(corpus, key = bool) if ele]
    
    #add start and stop sybmbols for each tweet
    for i in range(0,len(res)):
        res[i].insert(0,'<START>')
        res[i].append('<STOP>')
        for j in range(0,len(res[i])):
            if '@USER' in res[i][j]:
                res[i][j] = '@USER'
    
    predicted_tags = []
    #count = 0
    for i in range(0,len(res)):
        tweets = res[i] #sentences
        pis = {}
        for j in range(0,len(tweets)):
            observedOutput = tweets[j] #tokens
            
            #append predicted tags from pis to predicted_tags list at the end of every sentence
            if observedOutput =='<STOP>': 
                pisList = list(pis.values())
                for i in range(0,len(pisList)):
                    if i>0:
                        for j in list(pisList[i].keys()):
                            predicted_tags.append(j)
                pis = {}
                break

            #list of all possible tags for current token, if unseen token, empty list returned
            possibleTags = tokenTags(observedOutput, emission_prob) 
            if '<STOP>' in possibleTags:
                possibleTags.remove('<STOP>')
            
            pis[j] = {}
            temp = []
            if j == 1:
                if "@USER" in observedOutput:
                    tag = '@'
                    pis[j][tag] = 1
                    continue
                if "http" in observedOutput:
                    tag = 'U'
                    pis[j][tag] = 1
                    continue
                try:
                    if float(observedOutput):
                        tag = '$'
                        pis[j][tag] = 1
                        continue
                except:
                    pass
                        
                try:
                    for k in possibleTags:
                        #get all Pis from state 0 (start) to state 1
                        temp.append(transition_prob['<START>'][k] * emission_prob[k][observedOutput])

                    #getting the state with the highest probability in list temp
                    indexOfMax = temp.index(max(temp))
                    tag = possibleTags[indexOfMax]
                    #add the most likely state in pis
                    pis[j][tag] = max(temp)

                except:
                    try:
                        #append emission probability with different suffixes and get the one with max probabilty
                        possibleTags_suffix4 = tokenTags(observedOutput[-4:], emission_prob_suffix4)
                        temp4 = []
                        for k in possibleTags_suffix4:
                            temp4.append(transition_prob['<START>'][k] * emission_prob_suffix4[k][observedOutput[-4:]])
                    
                        possibleTags_suffix3 = tokenTags(observedOutput[-3:], emission_prob_suffix3)
                        temp3 = []
                        for k in possibleTags_suffix3:
                            temp3.append(transition_prob['<START>'][k] * emission_prob_suffix3[k][observedOutput[-3:]])
                        
                        possibleTags_suffix2 = tokenTags(observedOutput[-2:], emission_prob_suffix2)
                        temp2 = []
                        for k in possibleTags_suffix2:
                            temp2.append(transition_prob['<START>'][k] * emission_prob_suffix2[k][observedOutput[-2:]])
                        possibleTags_dict = {'possibleTags_suffix2': possibleTags_suffix2, 'possibleTags_suffix3': possibleTags_suffix3, 'possibleTags_suffix4': possibleTags_suffix4}
                        tempArr_dict = {'temp2': temp2, 'temp3': temp3, 'temp4':temp4}
                        
                        #getting the state with the highest probability in list temp
                        #temp = list(map(lambda x: max(x),[temp2,temp3,temp4]))
                        temp = [max(i) if i else 0 for i in [temp2,temp3,temp4]]
                        if max(temp) == 0:
                            raise Exception
                        indexOfMaxTemp = temp.index(max(temp))
                        indexOfMax = tempArr_dict['temp'+ str(indexOfMaxTemp +2)].index(temp[indexOfMaxTemp])
                        tag = possibleTags_dict['possibleTags_suffix' + str(indexOfMaxTemp +2)][indexOfMax]
                        #add the most likely state in pis
                        pis[j][tag] = max(temp)
                        
                    except:
                        if '<STOP>' in transition_prob['<START>'].keys():
                            new = transition_prob['<START>'].copy()
                            del new['<STOP>']
                            tag = list(new.keys())[list(new.values()).index(max(new.values()))]
                            pis[j][tag] = max(new.values())
                        else:
                            tag = list(transition_prob['<START>'].keys())[list(transition_prob['<START>'].values()).index(max(transition_prob['<START>'].values()))]
                            pis[j][tag] = max(transition_prob['<START>'].values())


            if j>1:
                previous_state = list(pis[j-1].keys())[0]
                temp =[]
                if "@USER" in observedOutput:
                    tag = '@'
                    pis[j][tag] = 1
                    continue
                if "http" in observedOutput:
                    tag = 'U'
                    pis[j][tag] = 1
                    continue
                try:
                    if float(observedOutput):
                        tag = '$'
                        pis[j][tag] = 1
                        continue
                except:
                    pass
                try:
                    for k in possibleTags:
                        #get all Pis from state 0 (start) to state 1
                        temp.append(transition_prob[previous_state][k] * emission_prob[k][observedOutput])

                    #getting the state with the highest probability in list temp
                    indexOfMax = temp.index(max(temp))
                    tag = possibleTags[indexOfMax]
                    #add the most likely state in pis
                    pis[j][tag] = max(temp)

                except:
                    try:
                        #append emission probability with different suffixes and get the one with max probabilty
                        possibleTags_suffix4 = tokenTags(observedOutput[-4:], emission_prob_suffix4)
                        temp4 = []
                        for k in possibleTags_suffix4:
                            temp4.append(transition_prob['<START>'][k] * emission_prob_suffix4[k][observedOutput[-4:]])
                        
                        possibleTags_suffix3 = tokenTags(observedOutput[-3:], emission_prob_suffix3)
                        temp3 = []
                        for k in possibleTags_suffix3:
                            temp3.append(transition_prob['<START>'][k] * emission_prob_suffix3[k][observedOutput[-3:]])
                        
                        possibleTags_suffix2 = tokenTags(observedOutput[-2:], emission_prob_suffix2)
                        temp2 = []
                        for k in possibleTags_suffix2:
                            temp2.append(transition_prob['<START>'][k] * emission_prob_suffix2[k][observedOutput[-2:]])
                        
                        possibleTags_dict = {'possibleTags_suffix2': possibleTags_suffix2, 'possibleTags_suffix3': possibleTags_suffix3, 'possibleTags_suffix4': possibleTags_suffix4}
                        tempArr_dict = {'temp2': temp2, 'temp3': temp3, 'temp4':temp4}
                        
                        #getting the state with the highest probability in list temp
                        temp = [max(i) if i else 0 for i in [temp2,temp3,temp4]]
                        if max(temp) == 0:
                            raise Exception
                        indexOfMaxTemp = temp.index(max(temp))
                        indexOfMax = tempArr_dict['temp'+ str(indexOfMaxTemp +2)].index(temp[indexOfMaxTemp])
                        tag = possibleTags_dict['possibleTags_suffix' + str(indexOfMaxTemp +2)][indexOfMax]
                        #add the most likely state in pis
                        pis[j][tag] = max(temp)

                    except:
                        if '<STOP>' in transition_prob[previous_state].keys():
                            new = transition_prob[previous_state].copy()
                            del new['<STOP>']
                            tag = list(new.keys())[list(new.values()).index(max(new.values()))]
                            pis[j][tag] =max(new.values())
                        else:
                            tag = list(transition_prob[previous_state].keys())[list(transition_prob[previous_state].values()).index(max(transition_prob[previous_state].values()))]
                            pis[j][tag] =max(transition_prob[previous_state].values())

    with open(out_predictions_filename, "w") as fhandle:
        for tags in predicted_tags:
            fhandle.write(f'{tags}\n')


# In[105]:


# in_tags = f'{ddir}/twitter_tags.txt'
# in_trans = f'{ddir}/trans_probs3.txt'
# in_emission = f'{ddir}/output_probs3.txt'
# in_test = f'{ddir}/twitter_dev_no_tag.txt'
# in_ans = f'{ddir}/twitter_dev_ans.txt'
# in_output = f'{ddir}/fb_predictions3.txt'


# In[106]:


# Right after initialisation
#viterbi_predict2(in_tags, in_trans, in_emission, in_test, in_output)


# In[107]:


# in_tags2 = f'{ddir}/twitter_tags.txt'
# in_trans2 = f'{ddir}/trans_probs4.txt'
# in_emission2 = f'{ddir}/output_probs4.txt'
# in_test2 = f'{ddir}/twitter_dev_no_tag.txt'
# in_ans2 = f'{ddir}/twitter_dev_ans.txt'
# in_output2 = f'{ddir}/fb_predictions4.txt'


# In[108]:


# After 10 iterations
#viterbi_predict2(in_tags2, in_trans2, in_emission2, in_test2, in_output2)


# In[109]:


# def evaluate(in_prediction_filename, in_answer_filename):
#     """Do not change this method"""
#     with open(in_prediction_filename) as fin:
#         predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

#     with open(in_answer_filename) as fin:
#         ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]
#     print(len(predicted_tags))
#     print(len(ground_truth_tags))
#     assert len(predicted_tags) == len(ground_truth_tags)
    
#     correct = 0
#     for pred, truth in zip(predicted_tags, ground_truth_tags):
#         if pred == truth: correct += 1
#     return correct, len(predicted_tags), correct/len(predicted_tags)


# In[110]:


#evaluate(in_output,in_ans)


# In[111]:


#evaluate(in_output2,in_ans)


# # Question 7

# In[112]:


in_train_filename = f'{ddir}/cat_price_changes_train.txt'
in_tag_filename = f'{ddir}/cat_states.txt'
out_trans_filename = f'{ddir}/cat_trans_probs.txt'
out_output_filename = f'{ddir}/cat_output_probs.txt'


# In[113]:


def forward_backward_cat(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh):
    random.seed(seed)
    file = in_train_filename
    #read test file
    file = open(file, encoding = "utf8")
    corpus = file.read().splitlines()

    tags = open(in_tag_filename,encoding = "utf8" )
    tags = tags.read().splitlines()
    
    #splitting input into sublist of tweets
    tweets = [list(sub) for ele, sub in groupby(corpus, key = bool) if ele]

    #add start and stop sybmbols for each tweet
    for i in range(0,len(tweets)):
        tweets[i].insert(0,'START_WORD')
        tweets[i].append('STOP_WORD')
    
    init_aMatrix = init_trans(in_tag_filename,seed)
    init_bMatrix = init_output(in_train_filename,in_tag_filename,seed)
    with open(f'{ddir}/trans_probs3.txt', 'w') as file:
        file.write(json.dumps(init_aMatrix))
    with open(f'{ddir}/output_probs3.txt', 'w') as file:
        file.write(json.dumps(init_bMatrix))
    
    #log_likelihoodArr = []
    curr = 0
    for j in range(max_iter):

        #print(j)
        gamma_list = []
        xi_list = []
        log_likelihood = 0
        
        for tweet in tweets:
            alpha = forward_alpha(init_aMatrix,init_bMatrix,tweet,tags)
            #print(alpha[len(tweet)-2]["<STOP>"])
           
            for i in alpha:
                if "<STOP>" in alpha[i]:
                    log_likelihood += math.log(alpha[i]["<STOP>"])
            
            beta = backward_beta(init_aMatrix,init_bMatrix,tweet,tags)
            gammaTerm = gamma(alpha,beta,tweet,tags)
            xiTerm = xi(alpha, beta, init_aMatrix,init_bMatrix,tweet,tags)
            gamma_list.append(gammaTerm)
            xi_list.append(xiTerm)

        init_aMatrix = update_trans(xi_list,tags)
        init_bMatrix = update_output(gamma_list,tweets,tags)

        if j > 0:
            if abs(log_likelihood - curr) < thresh:
                break
        #log_likelihoodArr.append(log_likelihood)
        curr = log_likelihood
        #print(log_likelihoodArr)
        
        
    with open(out_trans_filename, 'w') as file:
        file.write(json.dumps(init_aMatrix))
    with open(out_output_filename, 'w') as file:
        file.write(json.dumps(init_bMatrix))


# In[114]:


#forward_backward_cat(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,100000,8,1e-25)


# In[115]:


def cat_trans(trans_prob, state):
    max_prob = max(trans_prob[state].values())
    for j in trans_prob[state]:
        if trans_prob[state][j] == max_prob:
            return j


# In[116]:


def cat_output(emission_prob, state):
    max_prob = max(emission_prob[state].values())
    for j in emission_prob[state]:
        if emission_prob[state][j] == max_prob:
            return j


# In[117]:


def cat_viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    #read test file
    file = open(in_test_filename,encoding = "utf8")
    corpus = file.read()
    corpus = corpus.splitlines()
    #read tags file
    file = open(in_tags_filename,encoding = "utf8")
    tags = file.read()
    tags = tags.split()
    tags.append("<STOP>")
    tags.insert(0, "<START>")
    
    #load emission probs
    with open(in_output_probs_filename,encoding = "utf8") as json_file:
        emission_prob = json.load(json_file)
    #load transition probs
    with open(in_trans_probs_filename,encoding = "utf8") as json_file:
        transition_prob = json.load(json_file)
    
    #splitting input into sublist of tweets
    res = [list(sub) for ele, sub in groupby(corpus, key = bool) if ele]
    
    #add start and stop sybmbols for each tweet
    for i in range(0,len(res)):
        res[i].insert(0,'START_WORD')
        res[i].append('STOP_WORD')
    
    predicted_tags = []
    #count = 0
    for i in range(0,len(res)):
        tweets = res[i] #each stock seq
        pis = {}
        for j in range(0,len(tweets)):
            observedOutput = tweets[j] #each px change
            
            #append predicted tags from pis to predicted_tags list at the end of every sentence
            if observedOutput =='STOP_WORD':
                pisList = list(pis.values())
                #print(pisList)
                for i in range(0,len(pisList)):
                    if i>0:
                        state = list(pisList[i].keys())[-1]
                        next_state = cat_trans(transition_prob, state)
                        next_price = cat_output(emission_prob,  next_state)
                        predicted_tags.append(next_price)
                        break
                pis = {}
                break

            #list of all possible tags for current token, if unseen token, empty list returned
            possibleTags = tags 
            if '<STOP>' in possibleTags:
                possibleTags.remove('<STOP>')
            pis[j] = {}
            temp = []
           # print(transition_prob['<START>']['s0'] * emission_prob['s0']['0'])
            if j == 1:  
                for k in possibleTags[1:]:
                    #get all Pis from state 0 (start) to state 1
                    temp.append(transition_prob['<START>'][k] * emission_prob[k][observedOutput])

                #getting the state with the highest probability in list temp
                indexOfMax = temp.index(max(temp))
                tag = possibleTags[indexOfMax]
                #add the most likely state in pis
                pis[j][tag] = max(temp)

                if '<STOP>' in transition_prob['<START>'].keys():
                    new = transition_prob['<START>'].copy()
                    del new['<STOP>']
                    tag = list(new.keys())[list(new.values()).index(max(new.values()))]
                    pis[j][tag] = max(new.values())
                else:
                    tag = list(transition_prob['<START>'].keys())[list(transition_prob['<START>'].values()).index(max(transition_prob['<START>'].values()))]
                    pis[j][tag] = max(transition_prob['<START>'].values())

            if j>1:
                previous_state = list(pis[j-1].keys())[0]
                
                for k in possibleTags[1:]:
                    #get all Pis from state 0 (start) to state 1
                    temp.append(transition_prob[previous_state][k] * emission_prob[k][observedOutput])

                #getting the state with the highest probability in list temp
                indexOfMax = temp.index(max(temp))
                tag = possibleTags[indexOfMax]
                #add the most likely state in pis
                pis[j][tag] = max(temp)


                if '<STOP>' in transition_prob[previous_state].keys():
                    new = transition_prob[previous_state].copy()
                    del new['<STOP>']
                    tag = list(new.keys())[list(new.values()).index(max(new.values()))]

                    pis[j][tag] =max(new.values())
                    #print(pis[j][tag])
                else:
                    tag = list(transition_prob[previous_state].keys())[list(transition_prob[previous_state].values()).index(max(transition_prob[previous_state].values()))]
                    pis[j][tag] =max(transition_prob[previous_state].values())
                    #print(pis[j][tag])

    with open(out_predictions_filename, "w") as fhandle:
        for tags in predicted_tags:
            fhandle.write(f'{tags}\n')


# In[118]:


in_tags = f'{ddir}/cat_states.txt'
in_trans = f'{ddir}/cat_trans_probs.txt'
in_emission = f'{ddir}/cat_output_probs.txt'
in_test = f'{ddir}/cat_price_changes_dev.txt'
in_ans = f'{ddir}/cat_price_changes_dev_ans.txt'
in_output = f'{ddir}/cat_predictions.txt'


# In[119]:


#cat_viterbi_predict2(in_tags, in_trans, in_emission, in_test, in_output)


# In[120]:


#evaluate_ave_squared_error(in_output, in_ans)


# In[ ]:


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)

def evaluate_ave_squared_error(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    error = 0.0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        error += (int(pred) - int(truth))**2
    return error/len(predicted_tags), error, len(predicted_tags)

def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    ''' 

    ddir = 'C:/Users/lowxi/Documents/Y3 Sem 1/BT3102/projectfiles/Q6' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/twitter_train_no_tag.txt'
    in_tag_filename     = f'{ddir}/twitter_tags.txt'
    out_trans_filename  = f'{ddir}/trans_probs4.txt'
    out_output_filename = f'{ddir}/output_probs4.txt'
    max_iter = 10
    seed     = 8
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    trans_probs_filename3 =  f'{ddir}/trans_probs3.txt'
    output_probs_filename3 = f'{ddir}/output_probs3.txt'
    viterbi_predictions_filename3 = f'{ddir}/fb_predictions3.txt'
    viterbi_predict2_fb(in_tags_filename, trans_probs_filename3, output_probs_filename3, in_test_filename,
                     viterbi_predictions_filename3)
    correct, total, acc = evaluate(viterbi_predictions_filename3, in_ans_filename)
    print(f'iter 0 prediction accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename4 =  f'{ddir}/trans_probs4.txt'
    output_probs_filename4 = f'{ddir}/output_probs4.txt'
    viterbi_predictions_filename4 = f'{ddir}/fb_predictions4.txt'
    viterbi_predict2_fb(in_tags_filename, trans_probs_filename4, output_probs_filename4, in_test_filename,
                     viterbi_predictions_filename4)
    correct, total, acc = evaluate(viterbi_predictions_filename4, in_ans_filename)
    print(f'iter 10 prediction accuracy:   {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/cat_price_changes_train.txt'
    in_tag_filename     = f'{ddir}/cat_states.txt'
    out_trans_filename  = f'{ddir}/cat_trans_probs.txt'
    out_output_filename = f'{ddir}/cat_output_probs.txt'
    max_iter = 1000000
    seed     = 8
    thresh   = 1e-4
    forward_backward_cat(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    in_test_filename         = f'{ddir}/cat_price_changes_dev.txt'
    in_trans_probs_filename  = f'{ddir}/cat_trans_probs.txt'
    in_output_probs_filename = f'{ddir}/cat_output_probs.txt'
    in_states_filename       = f'{ddir}/cat_states.txt'
    predictions_filename     = f'{ddir}/cat_predictions.txt'
    cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                predictions_filename)

    in_ans_filename     = f'{ddir}/cat_price_changes_dev_ans.txt'
    ave_sq_err, sq_err, num_ex = evaluate_ave_squared_error(predictions_filename, in_ans_filename)
    print(f'average squared error for {num_ex} examples: {ave_sq_err}')

if __name__ == '__main__':
    run()

