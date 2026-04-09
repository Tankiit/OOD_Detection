import argparse
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Train model and expert predictors (probes).")
    parser.add_argument("--input_path", type=str, default="./extracted_states.pt", help="Path to the extracted .pt file.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength for logistic regression.")
    return parser.parse_args()

def build_arrays(input_path):
    """Reads saved tensors and builds features and target arrays."""
    data = torch.load(input_path)
    
    # Convert to NumPy for scikit-learn
    def to_np(t):
        return t.numpy() if torch.is_tensor(t) else t

    h_correct = to_np(data['h_correct'])
    h_wrong = to_np(data['h_wrong'])
    
    N_correct = len(h_correct)
    N_wrong = len(h_wrong)
    
    print(f"Found {N_correct} correct samples and {N_wrong} wrong samples.")
    
    # y_model: 1 for correct strings, 0 for wrong strings
    y_model = np.array([1] * N_correct + [0] * N_wrong)
    
    # X: Features (hidden states)
    X = np.concatenate([h_correct, h_wrong], axis=0)
    
    # y_expert: Concatenate expert correct/wrong targets
    y_exp_correct = to_np(data['y_expert_correct'])
    y_exp_wrong = to_np(data['y_expert_wrong'])
    y_expert = np.concatenate([y_exp_correct, y_exp_wrong], axis=0)
    
    return X, y_model, y_expert

def fit_probe(X, y, C=1.0):
    """Fits an L2-regularized logistic regression probe."""
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    sc = StandardScaler()
    
    X_scaled = sc.fit_transform(X)
    clf.fit(X_scaled, y)
    
    return clf, sc

def main():
    args = parse_args()
    
    print(f"Loading data from {args.input_path}...")
    X, y_model, y_expert = build_arrays(args.input_path)
    print(f"Data shape: X={X.shape}, y_model={y_model.shape}, y_expert={y_expert.shape}")
    
    # Simple train-test split (80/20) for demonstration
    # In a real scenario, you probably have predefined splits
    indices = np.arange(len(X))
    X_tr, X_te, y_model_tr, y_model_te, y_expert_tr, y_expert_te, idx_tr, idx_te = train_test_split(
        X, y_model, y_expert, indices, test_size=0.2, random_state=42
    )
    
    print("\n[1] Training f_pred (model correctness predictor)...")
    clf_pred, sc_pred = fit_probe(X_tr, y_model_tr, C=args.C)
    print(f"f_pred Accuracy (Train): {clf_pred.score(sc_pred.transform(X_tr), y_model_tr):.4f}")
    print(f"f_pred Accuracy (Test):  {clf_pred.score(sc_pred.transform(X_te), y_model_te):.4f}")
    
    print("\n[2] Training f_defer (expert predictor)...")
    clf_defer, sc_defer = fit_probe(X_tr, y_expert_tr, C=args.C)
    print(f"f_defer Accuracy (Train): {clf_defer.score(sc_defer.transform(X_tr), y_expert_tr):.4f}")
    print(f"f_defer Accuracy (Test):  {clf_defer.score(sc_defer.transform(X_te), y_expert_te):.4f}")
    
    print("\n[3] Computing deferral gap on test set...")
    # gap = logit_pred - logit_defer
    # A positive gap -> model log-odds higher than expert -> prefer model
    # A negative gap -> expert log-odds higher than model -> defer to expert
    logit_pred = clf_pred.decision_function(sc_pred.transform(X_te))
    logit_defer = clf_defer.decision_function(sc_defer.transform(X_te))
    
    gap = logit_pred - logit_defer
    print(f"Average gap on test set: {gap.mean():.4f}")
    
    defer_decision = gap < 0
    print(f"Defer to expert on: {defer_decision.mean()*100:.2f}% of test samples")

if __name__ == "__main__":
    main()
