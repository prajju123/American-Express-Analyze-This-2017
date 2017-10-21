#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:44:40 2017

@author: prajjwal
"""

training=pd.read_csv('train.csv')

del training["mvar1"]


del training["cm_key"]
training["card"]="0"
training.card[training.mvar49==1]=1
training.card[training.mvar50==1]=2
training.card[training.mvar51==1]=3
training["card"]=training["card"].astype('int')

train=training.drop(["mvar47","mvar48","mvar50","mvar51","mvar46","mvar49","mvar12"],axis=1)
x_train=train.drop(["card"],axis=1)

y_train=train["card"]
x_train.mvar39[x_train.mvar39==0]=1
x_train.mvar38[x_train.mvar38==0]=1
x_train.mvar37[x_train.mvar37==0]=1
x_train.mvar36[x_train.mvar36==0]=1
x_train["rat3"]=x_train["mvar39"]/x_train["mvar38"]
x_train["rat2"]=x_train["mvar38"]/x_train["mvar37"]
x_train["rat1"]=x_train["mvar37"]/x_train["mvar36"]


x_train.mvar31[x_train.mvar31==0]=1
x_train.mvar30[x_train.mvar30==0]=1
x_train.mvar29[x_train.mvar29==0]=1
x_train.mvar28[x_train.mvar28==0]=1
x_train["rat9"]=x_train["mvar31"]/x_train["mvar30"]
x_train["rat8"]=x_train["mvar30"]/x_train["mvar29"]
x_train["rat7"]=x_train["mvar29"]/x_train["mvar28"]


clf5 = xgb.XGBClassifier(max_depth=6, n_estimators=585, learning_rate=0.05,scale_pos_weight=2.5,subsample=0.9).fit(x_train, y_train)
testing=pd.read_csv('leader.csv')

del testing["mvar1"]

del testing["mvar12"]

test=testing.drop(["cm_key"],axis=1)

test.mvar39[test.mvar39==0]=1
test.mvar38[test.mvar38==0]=1
test.mvar37[test.mvar37==0]=1
test.mvar36[test.mvar36==0]=1
test["rat3"]=test["mvar39"]/test["mvar38"]
test["rat2"]=test["mvar38"]/test["mvar37"]
test["rat1"]=test["mvar37"]/test["mvar36"]



test.mvar31[test.mvar31==0]=1
test.mvar30[test.mvar30==0]=1
test.mvar29[test.mvar29==0]=1
test.mvar28[test.mvar28==0]=1
test["rat9"]=test["mvar31"]/test["mvar30"]
test["rat8"]=test["mvar30"]/test["mvar29"]
test["rat7"]=test["mvar29"]/test["mvar28"]

p=clf5.predict_proba(test)
pred11=[row[0] for row in p]
pred12=[row[1] for row in p]
pred13=[row[2] for row in p]
pred14=[row[3] for row in p]


ans=pd.DataFrame({'No':pred11,'Supp':pred12,'Elite':pred13,'Credit':pred14,'id':testing.cm_key})
ans['Max_val'] = ans[["Credit","Elite","Supp","No"]].max(axis=1)
ans['Max'] = ans[["Credit","Elite","Supp","No"]].idxmax(axis=1)

ans2=ans[ans.Max!="No"]
ans2 = ans2.sort_values(['No'], ascending=[True])
ans2=ans2[0:1000]
ans2 = ans2.sort_values(['Max_val'], ascending=[False])
sub=ans2[["id","Max"]]
sub.to_csv("_No__one_IITKharagpur_104.csv",index=False,header=False)
