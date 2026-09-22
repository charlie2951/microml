#include <stdio.h>
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

void mlp_init(MLPModel *model, const int *layer_sizes, int num_layers, int is_regression) {
    model->num_layers = num_layers;
    model->is_regression = is_regression;

    model->layer_sizes = m_new(int, num_layers);
    memcpy(model->layer_sizes, layer_sizes, num_layers * sizeof(int));

    int num_weight_matrices = num_layers - 1;
    model->weights = m_new(float*, num_weight_matrices);
    model->biases = m_new(float*, num_weight_matrices);
    model->vw = m_new(float*, num_weight_matrices);
    model->vb = m_new(float*, num_weight_matrices);
    model->activations = m_new(float*, num_layers);

    // Allocate Input layer activations (Layer 0)
    model->activations[0] = m_new(float, layer_sizes[0]);

    // Allocate and initialize layers 0 to N-1
    for (int l = 0; l < num_weight_matrices; l++) {
        int in_dim = layer_sizes[l];
        int out_dim = layer_sizes[l + 1];

        model->weights[l] = m_new(float, in_dim * out_dim);
        model->biases[l] = m_new0(float, out_dim);
        model->vw[l] = m_new0(float, in_dim * out_dim);
        model->vb[l] = m_new0(float, out_dim);
        model->activations[l + 1] = m_new(float, out_dim);

        // Xavier Uniform initialization per layer
        float limit = sqrtf(6.0f / (in_dim + out_dim));
        for (int i = 0; i < in_dim * out_dim; i++) {
            model->weights[l][i] = rand_uniform_mp(limit);
        }
    }
}

void mlp_free(MLPModel *model) {
    int num_weight_matrices = model->num_layers - 1;

    for (int l = 0; l < num_weight_matrices; l++) {
        int in_dim = model->layer_sizes[l];
        int out_dim = model->layer_sizes[l + 1];

        if (model->weights[l]) m_del(float, model->weights[l], in_dim * out_dim);
        if (model->biases[l]) m_del(float, model->biases[l], out_dim);
        if (model->vw[l]) m_del(float, model->vw[l], in_dim * out_dim);
        if (model->vb[l]) m_del(float, model->vb[l], out_dim);
        if (model->activations[l + 1]) m_del(float, model->activations[l + 1], out_dim);
    }

    if (model->activations[0]) m_del(float, model->activations[0], model->layer_sizes[0]);

    m_del(float*, model->weights, num_weight_matrices);
    m_del(float*, model->biases, num_weight_matrices);
    m_del(float*, model->vw, num_weight_matrices);
    m_del(float*, model->vb, num_weight_matrices);
    m_del(float*, model->activations, model->num_layers);
    m_del(int, model->layer_sizes, model->num_layers);
}

void mlp_forward(MLPModel *model, const float *x) {
    // 1. Copy input vector into Layer 0 activations
    memcpy(model->activations[0], x, model->layer_sizes[0] * sizeof(float));

    // 2. Propagate through all layers
    for (int l = 0; l < model->num_layers - 1; l++) {
        int in_dim = model->layer_sizes[l];
        int out_dim = model->layer_sizes[l + 1];
        int is_output_layer = (l == model->num_layers - 2);

        for (int j = 0; j < out_dim; j++) {
            float sum = model->biases[l][j];
            for (int i = 0; i < in_dim; i++) {
                sum += model->activations[l][i] * model->weights[l][i * out_dim + j];
            }
            // Hidden Layers use ReLU | Output Layer uses Linear activation
            model->activations[l + 1][j] = is_output_layer ? sum : (sum > 0.0f ? sum : 0.0f);
        }
    }

    // 3. Apply Softmax if Classification Mode
    if (!model->is_regression) {
        int out_layer = model->num_layers - 1;
        int out_dim = model->layer_sizes[out_layer];

        float max_val = -1e9f;
        for (int k = 0; k < out_dim; k++) {
            if (model->activations[out_layer][k] > max_val) {
                max_val = model->activations[out_layer][k];
            }
        }

        float exp_sum = 0.0f;
        for (int k = 0; k < out_dim; k++) {
            model->activations[out_layer][k] = expf(model->activations[out_layer][k] - max_val);
            exp_sum += model->activations[out_layer][k];
        }
        for (int k = 0; k < out_dim; k++) {
            model->activations[out_layer][k] /= exp_sum;
        }
    }
}

void mlp_fit(MLPModel *model, const float *X, const float *y, int n_samples, int epochs, float lr, float momentum) {
    int last_layer = model->num_layers - 1;
    int out_dim = model->layer_sizes[last_layer];

    // Allocate gradient delta buffers for each layer dynamically
    float **deltas = m_new(float*, model->num_layers);
    for (int l = 1; l < model->num_layers; l++) {
        deltas[l] = m_new(float, model->layer_sizes[l]);
    }

    int log_interval = epochs >= 10 ? epochs / 10 : 1;

    for (int ep = 0; ep < epochs; ep++) {
        float total_loss = 0.0f;
        int correct = 0;

        for (int s = 0; s < n_samples; s++) {
            const float *x = &X[s * model->layer_sizes[0]];

            // 1. Forward Pass
            mlp_forward(model, x);

            // 2. Output Layer Error Gradient calculation
            if (model->is_regression) {
                const float *target = &y[s * out_dim];
                for (int k = 0; k < out_dim; k++) {
                    float diff = model->activations[last_layer][k] - target[k];
                    deltas[last_layer][k] = 2.0f * diff; // Gradient of MSE
                    total_loss += diff * diff;
                }
            } else {
                int label = (int)y[s];
                for (int k = 0; k < out_dim; k++) {
                    deltas[last_layer][k] = model->activations[last_layer][k] - (k == label ? 1.0f : 0.0f);
                }
                float p_correct = model->activations[last_layer][label] > 1e-7f ? model->activations[last_layer][label] : 1e-7f;
                total_loss += -logf(p_correct);

                int pred_class = 0;
                float max_p = model->activations[last_layer][0];
                for (int k = 1; k < out_dim; k++) {
                    if (model->activations[last_layer][k] > max_p) {
                        max_p = model->activations[last_layer][k];
                        pred_class = k;
                    }
                }
                if (pred_class == label) correct++;
            }

            // 3. Backpropagate error gradients through all hidden layers (Reverse Loop)
            for (int l = last_layer - 1; l >= 1; l--) {
                int current_dim = model->layer_sizes[l];
                int next_dim = model->layer_sizes[l + 1];

                for (int j = 0; j < current_dim; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < next_dim; k++) {
                        sum += deltas[l + 1][k] * model->weights[l][j * next_dim + k];
                    }
                    // Derivative of ReLU
                    deltas[l][j] = (model->activations[l][j] > 0.0f) ? sum : 0.0f;
                }
            }

            // 4. Update Weights and Biases using SGD with Momentum
            for (int l = 0; l < model->num_layers - 1; l++) {
                int in_dim = model->layer_sizes[l];
                int layer_out_dim = model->layer_sizes[l + 1];

                for (int i = 0; i < in_dim; i++) {
                    for (int j = 0; j < layer_out_dim; j++) {
                        int idx = i * layer_out_dim + j;
                        float grad = deltas[l + 1][j] * model->activations[l][i];
                        model->vw[l][idx] = momentum * model->vw[l][idx] - lr * grad;
                        model->weights[l][idx] += model->vw[l][idx];
                    }
                }
                for (int j = 0; j < layer_out_dim; j++) {
                    model->vb[l][j] = momentum * model->vb[l][j] - lr * deltas[l + 1][j];
                    model->biases[l][j] += model->vb[l][j];
                }
            }
        }

        // Logging
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

    // Clean up gradient delta buffers
    for (int l = 1; l < model->num_layers; l++) {
        m_del(float, deltas[l], model->layer_sizes[l]);
    }
    m_del(float*, deltas, model->num_layers);
}

int mlp_predict(MLPModel *model, const float *x, float *probs) {
    mlp_forward(model, x);

    int out_layer = model->num_layers - 1;
    int out_dim = model->layer_sizes[out_layer];
    int max_idx = 0;
    float max_p = model->activations[out_layer][0];

    for (int k = 0; k < out_dim; k++) {
        if (probs != NULL) {
            probs[k] = model->activations[out_layer][k];
        }
        if (model->activations[out_layer][k] > max_p) {
            max_p = model->activations[out_layer][k];
            max_idx = k;
        }
    }
    return max_idx;
}

void mlp_predict_reg(MLPModel *model, const float *x, float *out) {
    mlp_forward(model, x);
    int out_layer = model->num_layers - 1;
    int out_dim = model->layer_sizes[out_layer];

    for (int k = 0; k < out_dim; k++) {
        out[k] = model->activations[out_layer][k];
    }
}

int mlp_save(const MLPModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) {
        return -1; // File creation failed
    }

    // Write file signature header for safety validation (Magic Bytes)
    const char magic[4] = {'M', 'L', 'P', '1'};
    if (fwrite(magic, sizeof(char), 4, f) != 4) {
        fclose(f);
        return -2;
    }

    // Write metadata
    fwrite(&model->num_layers, sizeof(int), 1, f);
    fwrite(&model->is_regression, sizeof(int), 1, f);
    fwrite(model->layer_sizes, sizeof(int), model->num_layers, f);

    // Write weights and biases for each connection layer
    int num_weight_matrices = model->num_layers - 1;
    for (int l = 0; l < num_weight_matrices; l++) {
        int in_dim = model->layer_sizes[l];
        int out_dim = model->layer_sizes[l + 1];

        fwrite(model->weights[l], sizeof(float), in_dim * out_dim, f);
        fwrite(model->biases[l], sizeof(float), out_dim, f);
    }

    fclose(f);
    return 0; // Success
}

int mlp_load(MLPModel *model, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        return -1; // File not found or read error
    }

    // Read and verify magic signature
    char magic[4];
    if (fread(magic, sizeof(char), 4, f) != 4 || 
        magic[0] != 'M' || magic[1] != 'L' || magic[2] != 'P' || magic[3] != '1') {
        fclose(f);
        return -2; // Invalid file format signature
    }

    // Read metadata header
    int num_layers = 0;
    int is_regression = 0;
    fread(&num_layers, sizeof(int), 1, f);
    fread(&is_regression, sizeof(int), 1, f);

    int *layer_sizes = m_new(int, num_layers);
    fread(layer_sizes, sizeof(int), num_layers, f);

    // Initialize allocations for the target structure model
    mlp_init(model, layer_sizes, num_layers, is_regression);
    m_del(int, layer_sizes, num_layers); // Temporarily freed since mlp_init clones layer_sizes internally

    // Read serialized weight and bias float arrays back into memory
    int num_weight_matrices = model->num_layers - 1;
    for (int l = 0; l < num_weight_matrices; l++) {
        int in_dim = model->layer_sizes[l];
        int out_dim = model->layer_sizes[l + 1];

        fread(model->weights[l], sizeof(float), in_dim * out_dim, f);
        fread(model->biases[l], sizeof(float), out_dim, f);
    }

    fclose(f);
    return 0; // Success
}
