# Equal Penalty Linear Regression
The equal penalty regression model sets a threshold in the loss function such that the loss beyond the threshold is almost constant. It is not set to be exactly constant as the gradient would vanish in such a case causing the model not to converge.

The equalpenalty_regression.py file requires the Streamlit library to be installed and be ready to use. All results are illustrated via Streamlit. 

For the You-Do model a class named EqualPenaltyReg which takes thresh, tolerance, max_iter, alpha, and l2_param parameters is built. Instantiating this class with l2_param=0. or l2_param={>0 VALUE} would result in non-regularized and regularized versions of the model, respectively.

The We-Do model is used as it is and the results are compared with those obtained from the You-Do model.
