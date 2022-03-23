## Serhat Tadik - 23.03.2022

import numpy as np
import numpy.linalg as la
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sympy import Piecewise


def z_score_train(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    list_means = []
    list_std = []
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        list_means.append(df_std[column].mean())
        list_std.append(df_std[column].std())
    df_std = df_std.clip(-3.0, 3.0)
    return df_std, list_means, list_std


def z_score_test(df, list_means, list_std):
    df_std = df.copy()
    idx = 0
    for column in df_std.columns:
        meann = list_means[idx]
        std = list_std[idx]
        df_std[column] = (df_std[column] - meann) / std
        idx += 1
    df_std = df_std.clip(-3.0, 3.0)
    return df_std


class EqualPenaltyReg(BaseEstimator):
    def __init__(self, thresh=234, tolerance=1e-10, max_iter=20000, alpha=1., l2_param=0.1):
        """
        thresh: The threshold beyond which the loss function is almost constant.
        tolerance: For early-stopping the gradient descent
        max_iter: Maximum number of steps in the gradient descent
        alpha: learning rate
        l2_param: L2 regularization parameter
        """
        self.thresh = thresh
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = None
        self.loss_history = None
        self.l2_param=l2_param

    def calc_loss(self, X, y):
        if la.norm(y - np.dot(X, self.beta)) <= self.thresh:
            return 0.5 * la.norm(y - np.dot(X, self.beta)) + self.l2_param*(self.beta**2).sum()
        else:
            return self.thresh/2 + 0.01*la.norm(y - np.dot(X, self.beta))-self.thresh*0.01 + self.l2_param*(self.beta**2).sum()

    def calc_gradient(self, X, y):
        cnt = 0
        if (la.norm(y - np.dot(X, self.beta))) <= self.thresh:
            cnt = 1
            grad = X.T @ (np.dot(X, self.beta) - y) / y.shape[0] + self.l2_param*2*self.beta
            flag=0
        else:
            grad = -0.01*np.ones((X.shape[1],1)) + self.l2_param*2*self.beta
            flag=1
        return grad, cnt, flag

    def fit(self, X, y):
        self.loss_history = []
        self.beta = np.random.rand(X.shape[1], 1)
        beta_prev = self.beta.copy()
        cntt = 0
        for i in range(self.max_iter):
            loss = self.calc_loss(X, y)
            self.loss_history.append(loss)
            grad, cnt, flag = self.calc_gradient(X, y)
            self.beta = self.beta - self.alpha * grad
            cntt += cnt
            if i!=0:
                if la.norm(self.beta - beta_prev) < self.tolerance:
                    st.write(f'Early stopped at iteration {i+1}.')
                    break
            if(i%np.round(self.max_iter/4) == np.round(self.max_iter/4)-1):
                if flag==0:
                    st.write("In between the local minima and the threshold. Moving towards the local minima.")
                else:
                    st.write("Beyond the threshold. Moving towards the threshold.")
            beta_prev = self.beta
        st.write(f"\n%d iterations for moving towards the threshold." %(i+1-cntt))
        st.write(f"%d iterations for moving towards local minima from the threshold.\n" %(cntt))
        if cntt == 0:
            st.warning("WARNING: The algorithm didn't converge. Try increasing either the threshold or the number of iterations!")
        return self

    def predict(self, X):
        if self.beta is None:
            raise Exception('Train first.')
        return np.dot(X, self.beta)
        pass

    def score(self, X, y):
        if self.beta is None:
            raise Exception('Train first.')
        return r2_score(y, np.dot(X, self.beta)), mean_squared_error(y, np.dot(X, self.beta))


cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
target = pd.DataFrame(cal_housing.target)
X = X[["MedInc"]]

#st.write(X)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=42)
#st.write(X_train)
X_train_standard, list_means, list_std = z_score_train(X_train)
X_train_standard.insert(0,"1",1.)
X_test_standard = z_score_test(X_test, list_means, list_std)
X_test_standard.insert(0,"1",1.)

## Visualize the Ideal Loss Function
st.header('Ideal Loss Function in 1-D')
equ_penalty = lambda delta,a,p: Piecewise((la.norm(a-p), la.norm(a-p) <= delta), (delta, True))
x = list(np.linspace(-3,3,1000))
y = []
for i in range(len(x)):
    y.append(equ_penalty(2.0,0,x[i]))
fig,ax = plt.subplots()
ax.plot(x,y,linewidth = 3)
ax.plot(-2,2,'r*',linewidth=4)
ax.plot(2,2,'r*', linewidth=4)
ax.set_title('Equal Penalty Loss')
ax.set_xlabel(r"""$\^y$ - y""")
ax.set_ylabel('Loss')
ax.grid('True')
st.pyplot(fig)

## Visualize the Modified Feasible Loss Function
st.header('Modified Loss Function in 1-D')
st.write("The loss function is modified such that the gradient "
         "does not become 0 beyond the threshold."
         " Please note that this is just the visualization of the"
         " desired loss function."
         " The modified function is almost convex. That is, it illustrates local"
         " convexity properties.")
almost_equ_penalty = lambda delta,a,p: Piecewise((la.norm(a-p), la.norm(a-p) <= delta), (delta-0.1+0.05*la.norm(a-p), True))
x = list(np.linspace(-3,3,1000))
y = []
for i in range(len(x)):
    y.append(almost_equ_penalty(2.0,0,x[i]))
fig,ax = plt.subplots()
ax.plot(x,y,linewidth = 3)
ax.plot(-2,2,'r*',linewidth=4)
ax.plot(2,2,'r*', linewidth=4)
ax.set_title('Almost Equal Penalty Loss')
ax.set_xlabel(r"""$\^y$ - y""")
ax.set_ylabel('Loss')
ax.grid('True')
st.pyplot(fig)
## Visualization Ended.

## Implementation of the Model w/o Regularization
st.subheader("Implementation of the Model w/o Regularization")
ep_reg = EqualPenaltyReg(thresh=200, max_iter=10000,l2_param=0.)
ep_reg.fit(X_train_standard.values, y_train.values)
r2_scr, mse = ep_reg.score(X_test_standard.values, y_test.values)
st.subheader("Results")
st.write("Mean Squared Testing Error: %.4f" % mse)
st.write("R2 - Testing Score: %.4f" % r2_scr)
st.write("Beta: %s" % ep_reg.beta)
st.write("Loss:")
my_expander = st.expander("Expand to see the loss", expanded=False)
with my_expander:
    st.write(ep_reg.loss_history)

df_out = pd.DataFrame(dict(x=X_test["MedInc"], y= ep_reg.beta[1][0]*X_test['MedInc']+ep_reg.beta[0][0]))
## Implementation of the Model w/o Regularization Ended.

## Implementation of the Model with Regularization
st.subheader("Implementation of the Model with Regularization")
st.write("I have concerns about the stability of the regularized version of the model."
         " The loss function when regularization term is added may show local concave characteristics.")
st.write("Please be aware that the threshold set for the model w/o regularization"
         " might cause a rather long convergence time for the regularized model."
         " If you come across a warning about convergence, try one of the followings:")
st.write("1- Re-run the program. Random initial choice of beta might speed up the convergence.")
st.write("2- Increase max_iter parameter of the model. This will give you a chance for convergence.")
st.write("3- Increase the threshold. This will let your model start closer to the threshold or even between the local"
         " minimum and the threshold.")

ep_reg2 = EqualPenaltyReg(thresh=255,max_iter=100000,alpha=1.,l2_param=st.slider("L2 Parameter",0.,0.1,0.04))
ep_reg2.fit(X_train_standard.values, y_train.values)
r2_scr, mse = ep_reg2.score(X_test_standard.values, y_test.values)
st.subheader("Results")
st.write("Mean Squared Testing Error: %.4f" % mse)
st.write("R2 - Tesing Score: %.4f" % r2_scr)
st.write("Beta: %s" % ep_reg2.beta)
st.write("Loss:")
my_expander = st.expander("Expand to see the loss", expanded=False)
with my_expander:
    st.write(ep_reg2.loss_history)

df_out2 = pd.DataFrame(dict(x=X_test["MedInc"], y= ep_reg2.beta[1][0]*X_test['MedInc']+ep_reg2.beta[0][0]))
## Implementation of the Model with Regularization Ended.

## Implementation of the We-Do Model
def wedo_model(x, y, lam, alpha=0.000001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)

    for i in range(1000):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum() + 2 * lam * beta[0]
        g_b1 = -2 * (x * (y - y_pred)).sum() + 2 * lam * beta[1]

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta
X_train_standard.drop("1",axis=1, inplace=True)
X_test.insert(0,"1",1.)

beta = wedo_model(X_train_standard.values, y_train.values, lam=st.slider("L2 Parameter for We-Do Model",0.,1000.,600.))
df_out3 = pd.DataFrame(dict(x=X_test["MedInc"], y= beta[1]*X_test['MedInc']+beta[0]))

wedo_r2, wedo_mse = r2_score(y_test, np.dot(X_test.values, beta[:,None])), mean_squared_error(y_test, np.dot(X_test.values, beta[:,None]))
st.subheader("Results")
st.write("Mean Squared Error: %.4f" % wedo_mse)
st.write("R2-score: %.4f" % wedo_r2)
st.write("Beta: %s" % beta)
## Implementation of the We-Do model Ended.

## Visualization of the Results
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_test["MedInc"], y=y_test[0], mode='markers', name='Test Data Points'))
fig.add_trace(go.Scatter(x=df_out['x'], y=df_out['y'], mode='lines', name='Regresssion w/o Regularization'))
fig.add_trace(go.Scatter(x=df_out2['x'], y=df_out2['y'], mode='lines', name='Regresssion with Regularization'))
fig.add_trace(go.Scatter(x=df_out3['x'], y=df_out3['y'], mode='lines', name='We-Do Model with Regularization'))
st.plotly_chart(fig, use_container_width=True)
## Visualization of the Results Ended.

## Conclusion
st.subheader("Conclusion")
st.write("As the resulting chart suggests, there is not much difference between You-Do and We-Do models."
         " The reason is that, both models ultimately converge to the same local minimum. "
         "You-Do model converges rather slowly since the gradient beyond the threshold is smaller than that of"
         " the We-Do model.")
## Conclusion Ended.
