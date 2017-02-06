# ML-assignment1
Logistic Regression: primal and dual implementation with polynomial kernel

## Data

- x matrix in hw1x.dat
- y vector in hw1y.dat

## Usage

`python main.py <FLAGS>`  <-- will work for all questions and plot all graphs.

FLAGS:
- `--normalize yes/no` : will normalize or not the X matrix (default to yes)
- `--use_sgd yes/no` : yes=use `sklearn.linear_model.SGDClassifier` no=`sklearn.linear_model.LogisticRegression` (default to yes)
- `--n_iter 10000` : number of iterations for Gradient Descent, or max number of iterations for Logistic Regression (default 10000)
- `--q1d` only produce plots for Q1.d) ie: logistic regression with L2 regularization
- `--q1f` only produce plots for Q1.f) ie: 5 x logistic regression on data applied to 5 gaussian basis functions
- `--q1g` only produce plots for Q1.g) ie: logistic regression on data applied to 25 gaussian basis functions with L2 regularization
- `--q2` only produce plots for Q2.c) ie: polynomial kernelized logistic regression.
