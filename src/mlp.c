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

void mlp_init(MLPModel *model, int input_dim, int hidden_dim, int output_dim, int is_regression) {
    model->input_dim = input_dim;
    model->hidden_dim = hidden_dim;
    model->output_dim = output_dim;
    model->is_regression = is_regression;

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
    if (model->w1) m_del(float, model->w1, model->input_dim * model->hidden_dim);
    if (model->b1) m_del(float, model->b1, model->hidden_dim);
    if (model->w2) m_del(float, model->w2, model->hidden_dim * model->output_dim);
    if (model->b2) m_del(float, model->b2, model->output_dim);

    if (model->vw1) m_del(float, model->vw1, model->input_dim * model->hidden_dim);
    if (model->vb1) m_del(float, model->vb1, model->hidden_dim);
    if (model->vw2) m_del(float, model->vw2, model->hidden_dim * model->output_dim);
    if (model->vb2) m_del(float, model->vb2, model->output_dim);

    if (model->h_act) m_del(float, model->h_act, model->hidden_dim);
    if (model->out_act) m_del(float, model->out_act, model->output_dim);
}

void mlp_forward(MLPModel *model, const float *x) {
    // Hidden Layer (ReLU)
    for (int j = 0; j < model->hidden_dim; j++) {
        float sum = model->b1[j];
        for (int i = 0; i < model->input_dim; i++) {
            sum += x[i] * model->w1[i * model->hidden_dim + j];
        }
        model->h_act[j] = sum > 0.0f ? sum : 0.0f;
    }

    // Output Layer (Linear)
    for (int k = 0; k < model->output_dim; k++) {
        float sum = model->b2[k];
        for (int j = 0; j < model->hidden_dim; j++) {
            sum += model->h_act[j] * model->w2[j * model->output_dim + k];
        }
        model->out_act[k] = sum;
    }

    // Apply Softmax only for classification
    if (!model->is_regression) {
        float max_val = -1e9f;
        for (int k = 0; k < model->output_dim; k++) {
            if (model->out_act[k] > max_val) max_val = model->out_act[k];
        }

        float exp_sum = 0.0f;
        for (int k = 0; k < model->output_dim; k++) {
            model->out_act[k] = expf(model->out_act[k] - max_val);
            exp_sum += model->out_act[k];
        }
        for (int k = 0; k < model->output_dim; k++) {
            model->out_act[k] /= exp_sum;
        }
    }
}

void mlp_fit(MLPModel *model, const float *X, const float *y, int n_samples, int epochs, float lr, float momentum) {
    float *dh = m_new(float, model->hidden_dim);
    float *dout = m_new(float, model->output_dim);

    int log_interval = epochs >= 10 ? epochs / 10 : 1;

    for (int ep = 0; ep < epochs; ep++) {
        float total_loss = 0.0f;
        int correct = 0;

        for (int s = 0; s < n_samples; s++) {
            const float *x = &X[s * model->input_dim];

            mlp_forward(model, x);

            if (model->is_regression) {
                // MSE Loss: gradient = 2 * (pred - target)
                const float *target = &y[s * model->output_dim];
                for (int k = 0; k < model->output_dim; k++) {
                    float diff = model->out_act[k] - target[k];
                    dout[k] = 2.0f * diff;
                    total_loss += diff * diff;
                }
            } else {
                // Cross-Entropy Loss
                int label = (int)y[s];
                for (int k = 0; k < model->output_dim; k++) {
                    dout[k] = model->out_act[k] - (k == label ? 1.0f : 0.0f);
                }
                float p_correct = model->out_act[label] > 1e-7f ? model->out_act[label] : 1e-7f;
                total_loss += -logf(p_correct);

                int pred_class = 0;
                float max_p = model->out_act[0];
                for (int k = 1; k < model->output_dim; k++) {
                    if (model->out_act[k] > max_p) {
                        max_p = model->out_act[k];
                        pred_class = k;
                    }
                }
                if (pred_class == label) correct++;
            }

            // Hidden gradient
            for (int j = 0; j < model->hidden_dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < model->output_dim; k++) {
                    sum += dout[k] * model->w2[j * model->output_dim + k];
                }
                dh[j] = (model->h_act[j] > 0.0f) ? sum : 0.0f;
            }

            // Update W2 & B2
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

            // Update W1 & B1
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
            float avg_loss = total_loss / n_samples;
            if (model->is_regression) {
                mp_printf(&mp_plat_print, "Epoch %d/%d - MSE Loss: %.6f\n", ep + 1, epochs, (double)avg_loss);
            } else {
                float accuracy = ((float)correct / n_samples) * 100.0f;
                mp_printf(&mp_plat_print, "Epoch %d/%d - Loss: %.4f - Accuracy: %.1f%%\n",
                          ep + 1, epochs, (double)avg_loss, (double)accuracy);
            }
        }
    }

    m_del(float, dh, model->hidden_dim);
    m_del(float, dout, model->output_dim);
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

void mlp_predict_reg(MLPModel *model, const float *x, float *out) {
    mlp_forward(model, x);
    for (int k = 0; k < model->output_dim; k++) {
        out[k] = model->out_act[k];
    }
}