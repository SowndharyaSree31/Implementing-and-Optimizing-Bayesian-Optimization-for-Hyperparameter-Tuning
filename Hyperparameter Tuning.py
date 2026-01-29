import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from scipy.stats import norm
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# 1. Define the Objective Function (The "Black Box")
def objective(params):
    """
    Evaluates XGBoost performance given hyperparameters.
    params: [learning_rate, n_estimators, max_depth]
    """
    learning_rate, n_estimators, max_depth = params
    model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # We want to maximize accuracy (or minimize negative accuracy)
    data = load_breast_cancer()
    score = cross_val_score(model, data.data, data.target, cv=3, scoring='accuracy').mean()
    return score

# 2. The Acquisition Function: Expected Improvement (EI)
def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian Process surrogate model.
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

# 3. Strategy for proposing the next point
def propose_next_candidate(X_sample, Y_sample, gpr, bounds, n_restarts=10):
    """
    Finds the best hyperparameter set to try next by maximizing the EI.
    """
    best_x = None
    max_ei = -1
    
    def min_obj(X):
        return -expected_improvement(X.reshape(-1, 3), X_sample, Y_sample, gpr).flatten()

    # Find the best optimum by starting from multiple random points
    for res in range(n_restarts):
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if -res.fun > max_ei:
            max_ei = -res.fun
            best_x = res.x
            
    return best_x

# 4. The Bayesian Optimization Loop
def bayesian_optimization(n_iterations, bounds):
    # Initial random samples (Start with 3 random configurations)
    X_sample = np.array([np.random.uniform(b[0], b[1], 3) for b in bounds]).T
    Y_sample = np.array([objective(x) for x in X_sample]).reshape(-1, 1)

    # Gaussian Process with Matern kernel (flexible for noisy functions)
    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)

    print(f"Iteration | Best Accuracy | Current Params")
    print("-" * 50)

    for i in range(n_iterations):
        # Update Surrogate Model
        gpr.fit(X_sample, Y_sample)

        # Suggest next hyperparameters
        x_next = propose_next_candidate(X_sample, Y_sample, gpr, bounds)
        
        # Evaluate objective function
        y_next = objective(x_next)

        # Update samples
        X_sample = np.vstack((X_sample, x_next))
        Y_sample = np.vstack((Y_sample, y_next))

        if i % 2 == 0:
            print(f"{i:9} | {np.max(Y_sample):.4f}        | {x_next}")

    return X_sample, Y_sample

# --- Execution ---
# Bounds: [learning_rate, n_estimators, max_depth]
search_bounds = np.array([[0.01, 0.3], [50, 300], [3, 10]])
X_history, Y_history = bayesian_optimization(n_iterations=15, bounds=search_bounds)

# Results Comparison
print(f"\nOptimization Complete.")
print(f"Best Score Found: {np.max(Y_history):.4f}")
best_idx = np.argmax(Y_history)
print(f"Best Hyperparameters: {X_history[best_idx]}")
