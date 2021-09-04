import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


class train_model():
    def __init__(self):
        pass
        self.train_model_process()



    def train_model_process(self):
        dataset = pd.read_csv("ai4i2020.csv")
        Y = dataset['Air temperature [K]']
        X = dataset.drop(columns=['UDI','Product ID','Type','Machine failure','Air temperature [K]'])

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        arr = scaler.fit_transform(X)

        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(arr, Y, test_size=25, random_state=345)



        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()

        regressor.fit(X_train, Y_train)

        r2 = regressor.score(X_test,Y_test)
        n = X_test.shape[0]

        p = X_test.shape[1]
        adj_r2_li = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        print('adj R2 Linear:',adj_r2_li)

        from sklearn.linear_model import Ridge,RidgeCV
        ridgecv = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=10, normalize=True)
        ridgecv.fit(X_train, Y_train)
        ridge_lr = Ridge(alpha=ridgecv.alpha_)
        ridge_lr.fit(X_train, Y_train)
        ridge_score = ridge_lr.score(X_test, Y_test)
        print('ridge_score:',ridge_score)


        from sklearn.linear_model import Lasso,  LassoCV
        lassocv = LassoCV(cv=10, max_iter=20000, normalize=True)
        lassocv.fit(X_train, Y_train)
        lasso = Lasso(alpha=lassocv.alpha_)
        lasso.fit(X_train, Y_train)
        lass_score = lasso.score(X_test,Y_test)
        print('lass_score:',lass_score)

        from sklearn.linear_model import ElasticNet, ElasticNetCV
        elastic = ElasticNetCV(alphas=None, cv=10)
        elastic.fit(X_train, Y_train)
        elastic_lr = ElasticNet(alpha=elastic.alpha_, l1_ratio=elastic.l1_ratio)
        elastic_lr.fit(X_train, Y_train)
        elastic_score = elastic_lr.score(X_test,Y_test)
        print('elastic_score:',elastic_score)
        model_all = {'adj R2 Linear':adj_r2_li,'ridge_score':ridge_score,'lass_score':lass_score,'elastic_score':elastic_score}
        Keymax = max(model_all, key=lambda x: model_all[x])

        if Keymax == 'adj R2 Linear':
            model_selection = regressor
        if Keymax == 'ridge_score':
            model_selection = ridge_lr
        if Keymax == 'lass_score':
            model_selection = lasso
        if Keymax == 'elastic_score':
            model_selection = elastic_lr

        pickle.dump(model_selection, open('tutor_model.pkl','wb'))
        model = pickle.load(open('tutor_model.pkl', 'rb'))
# Loading model to compare the results


class Predict_model():
    def __init__(self,new_features):
        self.new_features = new_features

    def preprocess_predict(a):
        dataset = pd.read_csv("ai4i2020.csv")
        Y1 = dataset['Air temperature [K]']
        X1 = dataset.drop(columns=['UDI', 'Product ID', 'Type', 'Machine failure', 'Air temperature [K]'])
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        arr = scaler.fit_transform(X1)

        actual = [a]
        df = pd.DataFrame(actual)
        #X_test = pd.concat(df,axis=0)
        scaled_val = scaler.transform(actual)
        return scaled_val

