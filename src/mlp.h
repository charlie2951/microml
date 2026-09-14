#ifndef MLP_H
#define MLP_H

typedef struct {
    int input_dim;
    int hidden_dim;
    int output_dim;

    float *w1;
    float *b1;
    float *w2;
    float *b2;

    float *vw1;
    float *vb1;
    float *vw2;
    float *vb2;

    float *h_act;
    float *out_act;
    int is_regression; // 0 for Classification, 1 for Regression
} MLPModel;

void mlp_init(MLPModel *model, int input_dim, int hidden_dim, int output_dim, int is_regression);
void mlp_free(MLPModel *model);
void mlp_forward(MLPModel *model, const float *x);
void mlp_fit(MLPModel *model, const float *X, const float *y, int n_samples, int epochs, float lr, float momentum);
void mlp_predict_reg(MLPModel *model, const float *x, float *out);

#endif // MLP_H
