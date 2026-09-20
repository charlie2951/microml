import ulab
from ulab import numpy as np

def confusion_matrix(y_true, y_pred, num_classes=None, labels=None, title="Confusion Matrix"):
    """
    Computes and prints a confusion matrix, then returns Accuracy, Precision, Recall, 
    Macro F1-score, and the raw ulab matrix.
    """
    y_true_arr = np.array(y_true, dtype=np.int16)
    y_pred_arr = np.array(y_pred, dtype=np.int16)
    
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length")
        
    if len(y_true_arr) == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}, np.zeros((0, 0), dtype=np.int16)

    # Automatically infer number of unique classes if not provided
    if num_classes is None:
        max_label = int(max(np.max(y_true_arr), np.max(y_pred_arr)))
        num_classes = max_label + 1

    # Fallback default text labels if none provided
    if labels is None:
        labels = [f"Class {i}" for i in range(num_classes)]

    # 1. Build Confusion Matrix in ulab: rows = true, cols = pred
    cm = np.zeros((num_classes, num_classes), dtype=np.int16)
    for t, p in zip(y_true_arr, y_pred_arr):
        cm[t, p] += 1

    # ==========================================================
    # COMPUTE GLOBAL ACCURACY
    # ==========================================================
    total_correct = sum([cm[i, i] for i in range(num_classes)])
    accuracy = total_correct / len(y_true_arr)

    # ==========================================================
    # PRINT CONFUSION MATRIX & ACCURACY
    # ==========================================================
    print(f"\n--- {title} ---")
    header_str = " ".join([f"{str(l):>10}" for l in labels])
    print(f"{'Actual \\ Pred':<15} | {header_str}")
    print("-" * (18 + 11 * num_classes))
    
    for i in range(num_classes):
        row_str = " ".join([f"{int(cm[i, j]):>10}" for j in range(num_classes)])
        print(f"{str(labels[i]):<15} | {row_str}")
    print("-" * (18 + 11 * num_classes))
    print(f"Accuracy: {accuracy * 100:.2f}% ({total_correct}/{len(y_true_arr)})")

    # 2. Compute per-class metrics (One-vs-Rest)
    precisions = []
    recalls = []
    f1_scores = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision_c + recall_c > 0:
            f1_c = 2.0 * (precision_c * recall_c) / (precision_c + recall_c)
        else:
            f1_c = 0.0

        precisions.append(precision_c)
        recalls.append(recall_c)
        f1_scores.append(f1_c)

    # 3. Compute Macro Averages across all classes
    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    macro_f1 = sum(f1_scores) / num_classes

    metrics = {
        'accuracy': accuracy,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1
    }

    return metrics, cm

#classification report

def classification_report(y_true, y_pred, labels=None, digits=4):
    """
    Builds and prints a comprehensive classification report using ulab.
    
    Parameters:
        y_true  : list or ulab.ndarray of ground truth labels
        y_pred  : list or ulab.ndarray of predicted labels
        labels  : list of string names for each class
        digits  : number of decimal places for formatting (default: 4)
    """
    y_true_arr = np.array(y_true, dtype=np.int16)
    y_pred_arr = np.array(y_pred, dtype=np.int16)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length")

    n_samples = len(y_true_arr)
    if n_samples == 0:
        print("Empty arrays provided.")
        return

    # 1. Determine number of classes & labels
    max_label = int(max(np.max(y_true_arr), np.max(y_pred_arr)))
    num_classes = max_label + 1

    if labels is None:
        labels = [f"Class {i}" for i in range(num_classes)]

    # 2. Build Confusion Matrix in ulab
    cm = np.zeros((num_classes, num_classes), dtype=np.int16)
    for t, p in zip(y_true_arr, y_pred_arr):
        cm[t, p] += 1

    # 3. Calculate Per-Class Metrics & Support
    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support = tp + fn  # Total true instances for class c

        precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision_c + recall_c > 0:
            f1_c = 2.0 * (precision_c * recall_c) / (precision_c + recall_c)
        else:
            f1_c = 0.0

        precisions.append(precision_c)
        recalls.append(recall_c)
        f1_scores.append(f1_c)
        supports.append(support)

    # 4. Global Accuracy
    total_correct = sum([cm[i, i] for i in range(num_classes)])
    accuracy = total_correct / n_samples

    # 5. Macro Averages (Unweighted mean across classes)
    macro_prec = sum(precisions) / num_classes
    macro_rec = sum(recalls) / num_classes
    macro_f1 = sum(f1_scores) / num_classes

    # 6. Weighted Averages (Weighted by class support)
    weighted_prec = sum([p * s for p, s in zip(precisions, supports)]) / n_samples
    weighted_rec = sum([r * s for r, s in zip(recalls, supports)]) / n_samples
    weighted_f1 = sum([f * s for f, s in zip(f1_scores, supports)]) / n_samples

    # ==========================================================
    # FORMATTED REPORT PRINTING
    # ==========================================================
    width = max([len(str(l)) for l in labels] + [12]) + 2
    fmt = f"{{:<{width}}} {{:>10}} {{:>10}} {{:>10}} {{:>10}}"
    line_len = width + 44

    print("\n" + "=" * line_len)
    print(" " * ((line_len - 21) // 2) + "CLASSIFICATION REPORT")
    print("=" * line_len)
    print(fmt.format("Class", "Precision", "Recall", "F1-Score", "Support"))
    print("-" * line_len)

    # Print individual classes
    for i in range(num_classes):
        label_str = str(labels[i])
        p_str = f"{precisions[i]:.{digits}f}"
        r_str = f"{recalls[i]:.{digits}f}"
        f_str = f"{f1_scores[i]:.{digits}f}"
        s_str = str(supports[i])
        print(fmt.format(label_str, p_str, r_str, f_str, s_str))

    print("-" * line_len)

    # Print overall accuracy row
    acc_str = f"{accuracy:.{digits}f}"
    print(fmt.format("Accuracy", "", "", acc_str, str(n_samples)))

    # Print Macro Average row
    macro_p_str = f"{macro_prec:.{digits}f}"
    macro_r_str = f"{macro_rec:.{digits}f}"
    macro_f_str = f"{macro_f1:.{digits}f}"
    print(fmt.format("Macro Avg", macro_p_str, macro_r_str, macro_f_str, str(n_samples)))

    # Print Weighted Average row
    weighted_p_str = f"{weighted_prec:.{digits}f}"
    weighted_r_str = f"{weighted_rec:.{digits}f}"
    weighted_f_str = f"{weighted_f1:.{digits}f}"
    print(fmt.format("Weighted Avg", weighted_p_str, weighted_r_str, weighted_f_str, str(n_samples)))
    print("=" * line_len + "\n")