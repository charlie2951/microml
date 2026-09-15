#ifndef MLP_H
#define MLP_H

#include "py/runtime.h"

typedef struct {
    int num_layers;       // Total number of layers (input + hidden layers + output)
    int *layer_sizes;     // Array containing dimensions for each layer e.g., [2, 16, 8, 1]
    int is_regression;    // 1 for Regression (MSE), 0 for Classification (Cross-Entropy)

    // Dynamic parameters (indexed by layer index l from 0 to num_layers - 2)
    float **weights;      // weights[l] array size: layer_sizes[l] * layer_sizes[l + 1]
    float **biases;       // biases[l] array size: layer_sizes[l + 1]
    float **vw;           // Momentum buffers for weights
    float **vb;           // Momentum buffers for biases

    // Dynamic activations (indexed by layer index l from 0 to num_layers - 1)
    float **activations;  // activations[l] array size: layer_sizes[l]
} MLPModel;

void mlp_init(MLPModel *model, const int *layer_sizes, int num_layers, int is_regression);
void mlp_free(MLPModel *model);
void mlp_forward(MLPModel *model, const float *x);
void mlp_fit(MLPModel *model, const float *X, const float *y, int n_samples, int epochs, float lr, float momentum);

// Prediction APIs
int mlp_predict(MLPModel *model, const float *x, float *probs);
void mlp_predict_reg(MLPModel *model, const float *x, float *out);

#endif // MLP_H