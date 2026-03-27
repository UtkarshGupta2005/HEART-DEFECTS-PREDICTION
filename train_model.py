import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import pickle

# Load dataset
df = pd.read_csv("framingham.csv")

# Cleaning
df.drop(columns=['education','currentSmoker'], inplace=True)
df.rename(columns={'male':'Sex'}, inplace=True)
df = df.drop_duplicates()

# Filtering
df = df[(df['age']>=20)&(df['age']<=100)]
df = df[(df['totChol']>100)]
df = df[df['BMI']>10]
df = df[df['glucose']>40]

# Fill missing
df.fillna(df.median(), inplace=True)

# Split
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train test
X_train,X_test,Y_train,Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42,stratify=Y)

# Model
rf = RandomForestClassifier(class_weight='balanced',n_jobs=-1,random_state=42)

search_space={
    'n_estimators':Integer(100,500),
    'max_depth':Integer(5,50),
    'min_samples_split':Integer(2,20),
    'min_samples_leaf':Integer(1,20),
    'max_features':Categorical(['sqrt','log2']),
    'bootstrap':Categorical([True,False])
}

bayes_opt=BayesSearchCV(
    estimator=rf,
    search_spaces=search_space,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
  )

bayes_opt.fit(X_train,Y_train)

best_rf=bayes_opt.best_estimator_
best_rf.fit(X_train,Y_train)

# Save model + scaler
pickle.dump(best_rf, open("model/model.pkl","wb"))
pickle.dump(scaler, open("model/scaler.pkl","wb"))

print("Model saved successfully")