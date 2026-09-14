#ifndef MICROML_H
#define MICROML_H

#include "py/runtime.h"
#include "py/misc.h"
#include <math.h>
#include <string.h>

// ==========================================
// 1. ENHANCED K-NEAREST NEIGHBORS (KNN)
// ==========================================
typedef struct {
    float *X;
    int *y;
    int n_samples;
    int n_features;
    int k;
    int n_classes;
} KNNModel;

static inline void knn_init(KNNModel *model, int k) {
    model->X = NULL;
    model->y = NULL;
    model->n_samples = 0;
    model->n_features = 0;
    model->k = k;
    model->n_classes = 2;
}

static inline void knn_fit(KNNModel *model, const float *X, const int *y, int n_samples, int n_features, int n_classes) {
    if (model->X) m_del(float, model->X, model->n_samples * model->n_features);
    if (model->y) m_del(int, model->y, model->n_samples);

    model->n_samples = n_samples;
    model->n_features = n_features;
    model->n_classes = n_classes;
    model->X = m_new(float, n_samples * n_features);
    model->y = m_new(int, n_samples);

    memcpy(model->X, X, n_samples * n_features * sizeof(float));
    memcpy(model->y, y, n_samples * sizeof(int));
}

typedef struct {
    float dist;
    int label;
} DistPair;

static inline int knn_predict_proba(const KNNModel *model, const float *x, float *probs) {
    if (model->n_samples == 0) return -1;

    DistPair *dists = m_new(DistPair, model->n_samples);

    for (int i = 0; i < model->n_samples; i++) {
        float sum = 0.0f;
        for (int j = 0; j < model->n_features; j++) {
            float diff = model->X[i * model->n_features + j] - x[j];
            sum += diff * diff;
        }
        dists[i].dist = sum;
        dists[i].label = model->y[i];
    }

    // Sort to find K nearest
    for (int i = 1; i < model->n_samples; i++) {
        DistPair key = dists[i];
        int j = i - 1;
        while (j >= 0 && dists[j].dist > key.dist) {
            dists[j + 1] = dists[j];
            j--;
        }
        dists[j + 1] = key;
    }

    int k_eff = (model->k < model->n_samples) ? model->k : model->n_samples;

    memset(probs, 0, model->n_classes * sizeof(float));
    for (int i = 0; i < k_eff; i++) {
        if (dists[i].label >= 0 && dists[i].label < model->n_classes) {
            probs[dists[i].label] += 1.0f / k_eff;
        }
    }

    m_del(DistPair, dists, model->n_samples);

    int best_class = 0;
    float max_p = probs[0];
    for (int c = 1; c < model->n_classes; c++) {
        if (probs[c] > max_p) {
            max_p = probs[c];
            best_class = c;
        }
    }
    return best_class;
}

// ==========================================
// 2. ENHANCED DECISION TREE
// ==========================================
typedef struct TreeNode {
    int feature_idx;
    float threshold;
    int left;
    int right;
    int label;
    float prob; // Confidence score for predicted class
} TreeNode;

typedef struct {
    TreeNode *nodes;
    int node_count;
    int max_depth;
    int n_features;
    int n_classes;
} DTModel;

static inline void dt_init(DTModel *model, int max_depth) {
    model->nodes = NULL;
    model->node_count = 0;
    model->max_depth = max_depth;
    model->n_features = 0;
    model->n_classes = 2;
}

static inline float calc_gini_multiclass(const int *y, const int *indices, int count, int n_classes) {
    if (count == 0) return 0.0f;
    int *counts = m_new0(int, n_classes);
    for (int i = 0; i < count; i++) {
        if (y[indices[i]] >= 0 && y[indices[i]] < n_classes) counts[y[indices[i]]]++;
    }
    float sum_p2 = 0.0f;
    for (int c = 0; c < n_classes; c++) {
        float p = (float)counts[c] / count;
        sum_p2 += p * p;
    }
    m_del(int, counts, n_classes);
    return 1.0f - sum_p2;
}

static inline int build_tree_node(DTModel *model, const float *X, const int *y, int *indices, int count, int depth) {
    int node_idx = model->node_count++;
    model->nodes = m_renew(TreeNode, model->nodes, model->node_count - 1, model->node_count);
    TreeNode *node = &model->nodes[node_idx];

    int *counts = m_new0(int, model->n_classes);
    for (int i = 0; i < count; i++) counts[y[indices[i]]]++;
    
    int best_c = 0;
    for (int c = 1; c < model->n_classes; c++) {
        if (counts[c] > counts[best_c]) best_c = c;
    }
    
    int leaf_label = best_c;
    float confidence = (float)counts[best_c] / count;
    m_del(int, counts, model->n_classes);

    if (depth >= model->max_depth || count <= 1 || confidence >= 0.99f) {
        node->feature_idx = -1;
        node->threshold = 0.0f;
        node->left = -1;
        node->right = -1;
        node->label = leaf_label;
        node->prob = confidence;
        return node_idx;
    }

    float best_gini = 1.0f;
    int best_feat = -1;
    float best_thresh = 0.0f;

    for (int f = 0; f < model->n_features; f++) {
        for (int i = 0; i < count; i++) {
            float thresh = X[indices[i] * model->n_features + f];
            int left_cnt = 0, right_cnt = 0;

            for (int j = 0; j < count; j++) {
                if (X[indices[j] * model->n_features + f] <= thresh) left_cnt++;
                else right_cnt++;
            }

            if (left_cnt == 0 || right_cnt == 0) continue;

            int *left_ind = m_new(int, left_cnt);
            int *right_ind = m_new(int, right_cnt);
            int l = 0, r = 0;

            for (int j = 0; j < count; j++) {
                if (X[indices[j] * model->n_features + f] <= thresh) left_ind[l++] = indices[j];
                else right_ind[r++] = indices[j];
            }

            float gini_l = calc_gini_multiclass(y, left_ind, left_cnt, model->n_classes);
            float gini_r = calc_gini_multiclass(y, right_ind, right_cnt, model->n_classes);
            float weighted_gini = (left_cnt * gini_l + right_cnt * gini_r) / count;

            m_del(int, left_ind, left_cnt);
            m_del(int, right_ind, right_cnt);

            if (weighted_gini < best_gini) {
                best_gini = weighted_gini;
                best_feat = f;
                best_thresh = thresh;
            }
        }
    }

    if (best_feat == -1) {
        node->feature_idx = -1;
        node->left = -1;
        node->right = -1;
        node->label = leaf_label;
        node->prob = confidence;
        return node_idx;
    }

    int l_cnt = 0, r_cnt = 0;
    for (int i = 0; i < count; i++) {
        if (X[indices[i] * model->n_features + best_feat] <= best_thresh) l_cnt++;
        else r_cnt++;
    }

    int *best_l_ind = m_new(int, l_cnt);
    int *best_r_ind = m_new(int, r_cnt);
    int l = 0, r = 0;

    for (int i = 0; i < count; i++) {
        if (X[indices[i] * model->n_features + best_feat] <= best_thresh) best_l_ind[l++] = indices[i];
        else best_r_ind[r++] = indices[i];
    }

    node->feature_idx = best_feat;
    node->threshold = best_thresh;
    node->label = leaf_label;
    node->prob = confidence;

    int left_child = build_tree_node(model, X, y, best_l_ind, l_cnt, depth + 1);
    int right_child = build_tree_node(model, X, y, best_r_ind, r_cnt, depth + 1);

    m_del(int, best_l_ind, l_cnt);
    m_del(int, best_r_ind, r_cnt);

    model->nodes[node_idx].left = left_child;
    model->nodes[node_idx].right = right_child;

    return node_idx;
}

static inline void dt_fit(DTModel *model, const float *X, const int *y, int n_samples, int n_features, int n_classes) {
    if (model->nodes) {
        m_del(TreeNode, model->nodes, model->node_count);
        model->nodes = NULL;
    }
    model->node_count = 0;
    model->n_features = n_features;
    model->n_classes = n_classes;

    int *indices = m_new(int, n_samples);
    for (int i = 0; i < n_samples; i++) indices[i] = i;

    build_tree_node(model, X, y, indices, n_samples, 0);

    m_del(int, indices, n_samples);
}

static inline int dt_predict(const DTModel *model, const float *x, float *confidence) {
    if (model->node_count == 0) return -1;

    int curr = 0;
    while (curr != -1) {
        TreeNode *node = &model->nodes[curr];
        if (node->left == -1 && node->right == -1) {
            if (confidence) *confidence = node->prob;
            return node->label;
        }
        if (x[node->feature_idx] <= node->threshold) {
            curr = node->left;
        } else {
            curr = node->right;
        }
    }
    return model->nodes[0].label;
}

// ==========================================
// 3. LINEAR & RBF KERNEL SVM
// ==========================================
#define SVM_KERNEL_LINEAR 0
#define SVM_KERNEL_RBF    1

typedef struct {
    float *weights;      // Linear SVM weights or Dual Coeffs for RBF
    float *support_vecs; // Only used for RBF
    float bias;
    int n_features;
    int n_samples;
    int kernel_type;
    float gamma;
} SVMModel;

static inline void svm_init(SVMModel *model, int kernel_type, float gamma) {
    model->weights = NULL;
    model->support_vecs = NULL;
    model->bias = 0.0f;
    model->n_features = 0;
    model->n_samples = 0;
    model->kernel_type = kernel_type;
    model->gamma = gamma;
}

static inline void svm_fit(SVMModel *model, const float *X, const int *y, int n_samples, int n_features, int epochs, float lr, float C) {
    if (model->weights) {
        m_del(float, model->weights, (model->kernel_type == SVM_KERNEL_LINEAR ? model->n_features : model->n_samples));
    }
    if (model->support_vecs) {
        m_del(float, model->support_vecs, model->n_samples * model->n_features);
    }

    model->n_features = n_features;
    model->n_samples = n_samples;
    model->bias = 0.0f;

    if (model->kernel_type == SVM_KERNEL_LINEAR) {
        model->weights = m_new0(float, n_features);

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n_samples; i++) {
                int target = (y[i] == 1) ? 1 : -1;
                float dot = 0.0f;
                for (int j = 0; j < n_features; j++) {
                    dot += model->weights[j] * X[i * n_features + j];
                }
                float margin = target * (dot + model->bias);

                if (margin < 1.0f) {
                    for (int j = 0; j < n_features; j++) {
                        model->weights[j] -= lr * ((2.0f / epochs * model->weights[j]) - (C * target * X[i * n_features + j]));
                    }
                    model->bias += lr * C * target;
                } else {
                    for (int j = 0; j < n_features; j++) {
                        model->weights[j] -= lr * (2.0f / epochs * model->weights[j]);
                    }
                }
            }
        }
    } else { // RBF Kernel Optimization
        model->weights = m_new0(float, n_samples);
        model->support_vecs = m_new(float, n_samples * n_features);
        memcpy(model->support_vecs, X, n_samples * n_features * sizeof(float));

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n_samples; i++) {
                int target = (y[i] == 1) ? 1 : -1;
                float sum = 0.0f;

                for (int sv = 0; sv < n_samples; sv++) {
                    float dist = 0.0f;
                    for (int j = 0; j < n_features; j++) {
                        float diff = X[sv * n_features + j] - X[i * n_features + j];
                        dist += diff * diff;
                    }
                    sum += model->weights[sv] * expf(-model->gamma * dist);
                }

                float margin = target * (sum + model->bias);
                if (margin < 1.0f) {
                    model->weights[i] += lr * (C * target - model->weights[i]);
                    model->bias += lr * C * target;
                }
            }
        }
    }
}

static inline int svm_predict(const SVMModel *model, const float *x) {
    if (model->n_features == 0) return -1;
    
    if (model->kernel_type == SVM_KERNEL_LINEAR) {
        float dot = 0.0f;
        for (int j = 0; j < model->n_features; j++) {
            dot += model->weights[j] * x[j];
        }
        return ((dot + model->bias) >= 0.0f) ? 1 : 0;
    } else { // RBF
        float sum = 0.0f;
        for (int i = 0; i < model->n_samples; i++) {
            float dist = 0.0f;
            for (int j = 0; j < model->n_features; j++) {
                float diff = model->support_vecs[i * model->n_features + j] - x[j];
                dist += diff * diff;
            }
            sum += model->weights[i] * expf(-model->gamma * dist);
        }
        return ((sum + model->bias) >= 0.0f) ? 1 : 0;
    }
}

#endif // MICROML_H