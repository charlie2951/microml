#ifndef MLP_H
#define MLP_H

#include "py/runtime.h"

typedef struct {
    int input_dim;
    int hidden_dim;
    int output_dim;

    // Parameters
    float *w1; // [input_dim * hidden_dim]
    float *b1; // [hidden_dim]
    float *w2; // [hidden_dim * output_dim]
    float *b2; // [output_dim]

    // Momentum buffers
    float *vw1;
    float *vb1;
    float *vw2;
    float *vb2;

    // Forward pass activations (cached for backward pass)
    float *h_act;   // [hidden_dim]
    float *out_act; // [output_dim]
} MLPModel;

void mlp_init(MLPModel *model, int input_dim, int hidden_dim, int output_dim);
void mlp_free(MLPModel *model);
void mlp_forward(MLPModel *model, const float *x);
void mlp_fit(MLPModel *model, const float *X, const int *y, int n_samples, int epochs, float lr, float momentum);
int mlp_predict(MLPModel *model, const float *x, float *probs);

#endif // MLP_H