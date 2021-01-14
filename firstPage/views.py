from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler,LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC 
# Create your views here.

def homepage(request):
	return render (request,'index.html')
 
n1 = n2 =n2 = n3 = n4 = n5 = n6 = n7 = n8 = n9 = n10 = n11 = n12 = n13 = n14 = n15 = n16 = n17 =n18=1
def result(request):
	global n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18
	data = pd.read_csv(r'C:/Users/INFOTEC/Desktop/Churn_customer/churn/models/Telco_customer_churn.csv')
	data= data.drop(['CustomerID','Churn Label','Churn Score','CLTV','Churn Reason'], axis=1)
	totalCharges = data.columns.get_loc("Total Charges")
	new_col = pd.to_numeric(data.iloc[:, totalCharges], errors='coerce')
    #coerce: the invalid parsing will be set as NaN
	data.iloc[:, totalCharges] = pd.Series(new_col)
	data = data.dropna(axis=0, subset=['Total Charges'])
	data = data.drop(['Country','State','Count','Gender','Lat Long','City','Zip Code','Lat Long','Latitude','Longitude'], axis=1)
	y = data.iloc[:,18]
	X = data.iloc[:,0:18]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
	categorical_features =data.select_dtypes(include=['object']).columns
	encoder= OrdinalEncoder()
	X_train[categorical_features]=encoder.fit_transform(X_train[categorical_features])
	X_test[categorical_features]=encoder.transform(X_test[categorical_features])
	Lnum = data.select_dtypes(include=['int64', 'float64']).columns
	X_train_svm,y_train_svm,X_test_svm,y_test_svm = X_train.copy(),y_train.copy(),X_test.copy(),y_test.copy()
	svm =SVC(C= 100, gamma= 0.01, kernel= 'rbf')
	svm.fit(X_train_svm, y_train_svm)
	if request.method=='POST':
		temp={}
		n1=request.POST.get('tenureMonths')
		n2=request.POST.get('monthlyCharges')
		n3=request.POST.get('totalCharges')
		n4_4=request.POST.get('seniorCitizen')
		if(n4_4=="no"):
			n4=0
		else:
			n4=1

		n5_5=request.POST.get('partner')
		if(n5_5=="no"):
			n5=0
		else:
			n5=1
		n6_6=request.POST.get('dependents')
		if(n6_6=="no"):
			n6=0
		else:
			n6=1
		
		n7_7=request.POST.get('phoneService')
		if(n7_7=="no"):
			n7=0
		else:
			n7=1
		n8_8=request.POST.get('multipleLines')
		if(n8_8=="no"):
			n8=0
		elif(n8_8=="yes"):
			n8=1
		else:
			n8=2

		n9_9=request.POST.get('internetService')
		if(n9_9=="no"):
			n9=0
		elif(n9_9=="DSL"):
			n9=1
		else:
			n9=2
		n10_10=request.POST.get('onlineSecurity')
		if(n10_10=="no"):
			n10=0
		elif(n10_10=="yes"):
			n10=1
		else:
			n10=2
	
		n11_11=request.POST.get('onlineBackup')
		if(n11_11=="no"):
			n11=0
		elif(n11_11=="yes"):
			n11=1
		else:
			n11=2
		
		n12_12=request.POST.get('deviceProtection')
		if(n12_12=="no"):
			n12=0
		elif(n12_12=="yes"):
			n12=1
		else:
			n12=2

		n13_13=request.POST.get('techSupport')
		if(n13_13=="no"):
			n13=0
		elif(n13_13=="yes"):
			n13=1
		else:
			n13=2
	
		n14_14=request.POST.get('streamingTV')
		if(n14_14=="no"):
			n14=0
		elif(n14_14=="yes"):
			n14=1
		else:
			n14=2
	
		n15_15=request.POST.get('streamingMovies')
		if(n15_15=="no"):
			n15=0
		elif(n15_15=="yes"):
			n15=1
		else:
			n15=2

		n16_16=request.POST.get('contract')
		if(n16_16=="Monthly"):
			n16=0
		elif(n16_16=="OneYear"):
			n16=1
		else:
			n16=2
		n17_17=request.POST.get('paperlessBilling')
		if(n17_17=="no"):
			n17=0
		else:
			n17=1
		
		n18_18=request.POST.get('paymentMethod')
		if(n18_18=="BankTransfer"):
			n18=3
		elif(n18_18=="Mailed"):
			n18=1
		else:
			n18=2
		n1=request.POST.get('tenureMonths')
	#testDTaa=pd.DataFrame({'x':temp}).transpose()
	pred=svm.predict([[n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18]])
	print(pred)
	scoreval="test with value"
	if pred==0:
		scoreval="churn"
	else:
		scoreval="no churn"

	#print(testDTaa)
	return render(request,'index.html',{'result':scoreval})
  