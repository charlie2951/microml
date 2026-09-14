#ifndef MICROML_H
#define MICROML_H

#include "py/runtime.h"
#include "py/misc.h"
#include <math.h>
#include <string.h>

// ==========================================
// 1. KNN REGRESSOR
// ==========================================
typedef struct {
    float *X;
    float *y;
    int n_samples;
    int n_features;
    int k;
} KNNRegressorModel;

static inline void knn_reg_init(KNNRegressorModel *model, int k) {
    model->X = NULL;
    model->y = NULL;
    model->n_samples = 0;
    model->n_features = 0;
    model->k = k;
}

static inline void knn_reg_fit(KNNRegressorModel *model, const float *X, const float *y, int n_samples, int n_features) {
    if (model->X) m_del(float, model->X, model->n_samples * model->n_features);
    if (model->y) m_del(float, model->y, model->n_samples);

    model->n_samples = n_samples;
    model->n_features = n_features;
    model->X = m_new(float, n_samples * n_features);
    model->y = m_new(float, n_samples);

    memcpy(model->X, X, n_samples * n_features * sizeof(float));
    memcpy(model->y, y, n_samples * sizeof(float));
}

typedef struct {
    float dist;
    float val;
} DistRegPair;

static inline float knn_reg_predict(const KNNRegressorModel *model, const float *x) {
    if (model->n_samples == 0) return 0.0f;

    DistRegPair *dists = m_new(DistRegPair, model->n_samples);

    for (int i = 0; i < model->n_samples; i++) {
        float sum = 0.0f;
        for (int j = 0; j < model->n_features; j++) {
            float diff = model->X[i * model->n_features + j] - x[j];
            sum += diff * diff;
        }
        dists[i].dist = sum;
        dists[i].val = model->y[i];
    }

    // Sort to find K nearest
    for (int i = 1; i < model->n_samples; i++) {
        DistRegPair key = dists[i];
        int j = i - 1;
        while (j >= 0 && dists[j].dist > key.dist) {
            dists[j + 1] = dists[j];
            j--;
        }
        dists[j + 1] = key;
    }

    int k_eff = (model->k < model->n_samples) ? model->k : model->n_samples;
    float pred_sum = 0.0f;
    for (int i = 0; i < k_eff; i++) {
        pred_sum += dists[i].val;
    }

    m_del(DistRegPair, dists, model->n_samples);
    return pred_sum / k_eff;
}

// ==========================================
// 2. DECISION TREE REGRESSOR (MSE Loss)
// ==========================================
typedef struct DTRegNode {
    int feature_idx;
    float threshold;
    int left;
    int right;
    float val;
} DTRegNode;

typedef struct {
    DTRegNode *nodes;
    int node_count;
    int max_depth;
    int n_features;
} DTRegressorModel;

static inline void dt_reg_init(DTRegressorModel *model, int max_depth) {
    model->nodes = NULL;
    model->node_count = 0;
    model->max_depth = max_depth;
    model->n_features = 0;
}

static inline float calc_mse(const float *y, const int *indices, int count) {
    if (count == 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < count; i++) sum += y[indices[i]];
    float mean = sum / count;

    float mse = 0.0f;
    for (int i = 0; i < count; i++) {
        float diff = y[indices[i]] - mean;
        mse += diff * diff;
    }
    return mse / count;
}

static inline int build_tree_reg_node(DTRegressorModel *model, const float *X, const float *y, int *indices, int count, int depth) {
    int node_idx = model->node_count++;
    model->nodes = m_renew(DTRegNode, model->nodes, model->node_count - 1, model->node_count);
    DTRegNode *node = &model->nodes[node_idx];

    float sum = 0.0f;
    for (int i = 0; i < count; i++) sum += y[indices[i]];
    float leaf_val = sum / count;

    if (depth >= model->max_depth || count <= 1) {
        node->feature_idx = -1;
        node->threshold = 0.0f;
        node->left = -1;
        node->right = -1;
        node->val = leaf_val;
        return node_idx;
    }

    float best_mse = 1e9f;
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

            float mse_l = calc_mse(y, left_ind, left_cnt);
            float mse_r = calc_mse(y, right_ind, right_cnt);
            float weighted_mse = (left_cnt * mse_l + right_cnt * mse_r) / count;

            m_del(int, left_ind, left_cnt);
            m_del(int, right_ind, right_cnt);

            if (weighted_mse < best_mse) {
                best_mse = weighted_mse;
                best_feat = f;
                best_thresh = thresh;
            }
        }
    }

    if (best_feat == -1) {
        node->feature_idx = -1;
        node->left = -1;
        node->right = -1;
        node->val = leaf_val;
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
    node->val = leaf_val;

    int left_child = build_tree_reg_node(model, X, y, best_l_ind, l_cnt, depth + 1);
    int right_child = build_tree_reg_node(model, X, y, best_r_ind, r_cnt, depth + 1);

    m_del(int, best_l_ind, l_cnt);
    m_del(int, best_r_ind, r_cnt);

    model->nodes[node_idx].left = left_child;
    model->nodes[node_idx].right = right_child;

    return node_idx;
}

static inline void dt_reg_fit(DTRegressorModel *model, const float *X, const float *y, int n_samples, int n_features) {
    if (model->nodes) {
        m_del(DTRegNode, model->nodes, model->node_count);
        model->nodes = NULL;
    }
    model->node_count = 0;
    model->n_features = n_features;

    int *indices = m_new(int, n_samples);
    for (int i = 0; i < n_samples; i++) indices[i] = i;

    build_tree_reg_node(model, X, y, indices, n_samples, 0);

    m_del(int, indices, n_samples);
}

static inline float dt_reg_predict(const DTRegressorModel *model, const float *x) {
    if (model->node_count == 0) return 0.0f;

    int curr = 0;
    while (curr != -1) {
        DTRegNode *node = &model->nodes[curr];
        if (node->left == -1 && node->right == -1) {
            return node->val;
        }
        if (x[node->feature_idx] <= node->threshold) {
            curr = node->left;
        } else {
            curr = node->right;
        }
    }
    return model->nodes[0].val;
}

// ==========================================
// 3. SUPPORT VECTOR REGRESSOR (SVR Linear)
// ==========================================
typedef struct {
    float *weights;
    float bias;
    int n_features;
} SVRModel;

static inline void svr_init(SVRModel *model) {
    model->weights = NULL;
    model->bias = 0.0f;
    model->n_features = 0;
}

static inline void svr_fit(SVRModel *model, const float *X, const float *y, int n_samples, int n_features, int epochs, float lr, float epsilon) {
    if (model->weights) {
        m_del(float, model->weights, model->n_features);
    }

    model->n_features = n_features;
    model->weights = m_new0(float, n_features);
    model->bias = 0.0f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < n_samples; i++) {
            float pred = model->bias;
            for (int j = 0; j < n_features; j++) {
                pred += model->weights[j] * X[i * n_features + j];
            }

            float error = pred - y[i];

            if (fabsf(error) > epsilon) {
                float sign = (error > 0.0f) ? 1.0f : -1.0f;
                for (int j = 0; j < n_features; j++) {
                    model->weights[j] -= lr * sign * X[i * n_features + j];
                }
                model->bias -= lr * sign;
            }
        }
    }
}

static inline float svr_predict(const SVRModel *model, const float *x) {
    if (model->n_features == 0) return 0.0f;
    float pred = model->bias;
    for (int j = 0; j < model->n_features; j++) {
        pred += model->weights[j] * x[j];
    }
    return pred;
}

#endif // MICROML_H
