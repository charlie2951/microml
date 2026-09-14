#include <math.h>
#include <string.h>
#include "py/runtime.h"
#include "py/mphal.h"
#include "mlp.h"

static float rand_uniform_mp(float limit) {
    #if MICROPY_PY_UTIME_TICKS_PERIOD
        uint32_t r = (uint32_t)mp_hal_ticks_ms();
    #else
        uint32_t r = (uint32_t)mp_hal_ticks_cpu();
    #endif

    r ^= r << 13;
    r ^= r >> 17;
    r ^= r << 5;

    float norm = (float)(r % 10000) / 10000.0f;
    return norm * 2.0f * limit - limit;
}

void mlp_init(MLPModel *model, int input_dim, int hidden_dim, int output_dim) {
    model->input_dim = input_dim;
    model->hidden_dim = hidden_dim;
    model->output_dim = output_dim;

    model->w1 = m_new(float, input_dim * hidden_dim);
    model->b1 = m_new0(float, hidden_dim);
    model->w2 = m_new(float, hidden_dim * output_dim);
    model->b2 = m_new0(float, output_dim);

    model->vw1 = m_new0(float, input_dim * hidden_dim);
    model->vb1 = m_new0(float, hidden_dim);
    model->vw2 = m_new0(float, hidden_dim * output_dim);
    model->vb2 = m_new0(float, output_dim);

    model->h_act = m_new(float, hidden_dim);
    model->out_act = m_new(float, output_dim);

    float limit1 = sqrtf(6.0f / (input_dim + hidden_dim));
    for (int i = 0; i < input_dim * hidden_dim; i++) {
        model->w1[i] = rand_uniform_mp(limit1);
    }

    float limit2 = sqrtf(6.0f / (hidden_dim + output_dim));
    for (int i = 0; i < hidden_dim * output_dim; i++) {
        model->w2[i] = rand_uniform_mp(limit2);
    }
}

void mlp_free(MLPModel *model) {
    if (model->w1) m_free(model->w1, model->input_dim * model->hidden_dim * sizeof(float));
    if (model->b1) m_free(model->b1, model->hidden_dim * sizeof(float));
    if (model->w2) m_free(model->w2, model->hidden_dim * model->output_dim * sizeof(float));
    if (model->b2) m_free(model->b2, model->output_dim * sizeof(float));

    if (model->vw1) m_free(model->vw1, model->input_dim * model->hidden_dim * sizeof(float));
    if (model->vb1) m_free(model->vb1, model->hidden_dim * sizeof(float));
    if (model->vw2) m_free(model->vw2, model->hidden_dim * model->output_dim * sizeof(float));
    if (model->vb2) m_free(model->vb2, model->output_dim * sizeof(float));

    if (model->h_act) m_free(model->h_act, model->hidden_dim * sizeof(float));
    if (model->out_act) m_free(model->out_act, model->output_dim * sizeof(float));
}

void mlp_forward(MLPModel *model, const float *x) {
    // 1. Input -> Hidden (Linear + ReLU)
    for (int j = 0; j < model->hidden_dim; j++) {
        float sum = model->b1[j];
        for (int i = 0; i < model->input_dim; i++) {
            sum += x[i] * model->w1[i * model->hidden_dim + j];
        }
        model->h_act[j] = sum > 0.0f ? sum : 0.0f; // ReLU
    }

    // 2. Hidden -> Output (Linear Identity Activation for Regression)
    for (int k = 0; k < model->output_dim; k++) {
        float sum = model->b2[k];
        for (int j = 0; j < model->hidden_dim; j++) {
            sum += model->h_act[j] * model->w2[j * model->output_dim + k];
        }
        model->out_act[k] = sum; // Linear output
    }
}

void mlp_fit(MLPModel *model, const float *X, const float *y, int n_samples, int epochs, float lr, float momentum) {
    float *dh = m_new(float, model->hidden_dim);
    float *dout = m_new(float, model->output_dim);

    int log_interval = epochs >= 10 ? epochs / 10 : 1;

    for (int ep = 0; ep < epochs; ep++) {
        float total_mse = 0.0f;

        for (int s = 0; s < n_samples; s++) {
            const float *x = &X[s * model->input_dim];
            const float *target = &y[s * model->output_dim];

            // Forward Pass
            mlp_forward(model, x);

            // Compute MSE Loss Gradient: dL/dOut = (y_pred - y_true)
            for (int k = 0; k < model->output_dim; k++) {
                float diff = model->out_act[k] - target[k];
                dout[k] = diff;
                total_mse += diff * diff;
            }

            // Hidden Gradient (Backprop through W2 and ReLU derivative)
            for (int j = 0; j < model->hidden_dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < model->output_dim; k++) {
                    sum += dout[k] * model->w2[j * model->output_dim + k];
                }
                dh[j] = (model->h_act[j] > 0.0f) ? sum : 0.0f;
            }

            // Update W2 and Bias 2
            for (int j = 0; j < model->hidden_dim; j++) {
                for (int k = 0; k < model->output_dim; k++) {
                    int idx = j * model->output_dim + k;
                    float grad = dout[k] * model->h_act[j];
                    model->vw2[idx] = momentum * model->vw2[idx] - lr * grad;
                    model->w2[idx] += model->vw2[idx];
                }
            }
            for (int k = 0; k < model->output_dim; k++) {
                model->vb2[k] = momentum * model->vb2[k] - lr * dout[k];
                model->b2[k] += model->vb2[k];
            }

            // Update W1 and Bias 1
            for (int i = 0; i < model->input_dim; i++) {
                for (int j = 0; j < model->hidden_dim; j++) {
                    int idx = i * model->hidden_dim + j;
                    float grad = dh[j] * x[i];
                    model->vw1[idx] = momentum * model->vw1[idx] - lr * grad;
                    model->w1[idx] += model->vw1[idx];
                }
            }
            for (int j = 0; j < model->hidden_dim; j++) {
                model->vb1[j] = momentum * model->vb1[j] - lr * dh[j];
                model->b1[j] += model->vb1[j];
            }
        }

        if ((ep + 1) % log_interval == 0 || ep == epochs - 1) {
            float mean_mse = total_mse / (n_samples * model->output_dim);
            mp_printf(&mp_plat_print, "Epoch %d/%d - MSE Loss: %.6f\n", ep + 1, epochs, (double)mean_mse);
        }
    }

    m_free(dh, model->hidden_dim * sizeof(float));
    m_free(dout, model->output_dim * sizeof(float));
}

void mlp_predict(MLPModel *model, const float *x, float *out_pred) {
    mlp_forward(model, x);
    for (int k = 0; k < model->output_dim; k++) {
        out_pred[k] = model->out_act[k];
    }
}
