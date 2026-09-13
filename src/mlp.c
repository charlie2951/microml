#include <math.h>
#include <string.h>
#include "py/runtime.h"
#include "py/mphal.h"
#include "mlp.h"

// MicroPython-compatible uniform random generator with fallback
static float rand_uniform_mp(float limit) {
    #if MICROPY_PY_UTIME_TICKS_PERIOD
        uint32_t r = (uint32_t)mp_hal_ticks_ms();
    #else
        uint32_t r = (uint32_t)mp_hal_ticks_cpu();
    #endif

    // Simple pseudo-random xorshift
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

    // Allocate on MicroPython GC Heap
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

    // Xavier initialization bounds
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

    // 2. Hidden -> Output (Linear + Softmax)
    float max_val = -1e9f;
    for (int k = 0; k < model->output_dim; k++) {
        float sum = model->b2[k];
        for (int j = 0; j < model->hidden_dim; j++) {
            sum += model->h_act[j] * model->w2[j * model->output_dim + k];
        }
        model->out_act[k] = sum;
        if (sum > max_val) max_val = sum; // Numerical stability
    }

    // Softmax normalization
    float exp_sum = 0.0f;
    for (int k = 0; k < model->output_dim; k++) {
        model->out_act[k] = expf(model->out_act[k] - max_val);
        exp_sum += model->out_act[k];
    }
    for (int k = 0; k < model->output_dim; k++) {
        model->out_act[k] /= exp_sum;
    }
}

void mlp_fit(MLPModel *model, const float *X, const int *y, int n_samples, int epochs, float lr, float momentum) {
    float *dh = m_new(float, model->hidden_dim);
    float *dout = m_new(float, model->output_dim);

    for (int ep = 0; ep < epochs; ep++) {
        for (int s = 0; s < n_samples; s++) {
            const float *x = &X[s * model->input_dim];
            int label = y[s];

            // Forward
            mlp_forward(model, x);

            // Output gradient (Softmax + Cross Entropy derivative)
            for (int k = 0; k < model->output_dim; k++) {
                dout[k] = model->out_act[k] - (k == label ? 1.0f : 0.0f);
            }

            // Hidden gradient (Backprop through W2 and ReLU derivative)
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
    }

    m_free(dh, model->hidden_dim * sizeof(float));
    m_free(dout, model->output_dim * sizeof(float));
}

int mlp_predict(MLPModel *model, const float *x, float *probs) {
    mlp_forward(model, x);

    int max_idx = 0;
    float max_p = model->out_act[0];

    for (int k = 0; k < model->output_dim; k++) {
        if (probs != NULL) {
            probs[k] = model->out_act[k];
        }
        if (model->out_act[k] > max_p) {
            max_p = model->out_act[k];
            max_idx = k;
        }
    }
    return max_idx;
}