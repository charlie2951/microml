#include <stdio.h>
#include <string.h>
#include "py/runtime.h"
#include "py/stream.h"
#include "cnn.h"

void cnn_init(CNNModel *model, int in_h, int in_w, int in_c, 
              int out_channels, int kernel_size, int pool_size,
              const int *dense_layers, int num_dense_layers) {
    
    model->in_h = in_h;
    model->in_w = in_w;
    model->in_c = in_c;

    // 1. Setup Conv2D Params
    model->conv.in_channels = in_c;
    model->conv.out_channels = out_channels;
    model->conv.kernel_size = kernel_size;
    model->conv.stride = 1;
    
    int w_size = out_channels * in_c * kernel_size * kernel_size;
    model->conv.weights = m_new0(float, w_size);
    model->conv.biases = m_new0(float, out_channels);

    // 2. Setup MaxPool2D Params
    model->pool.pool_size = pool_size;
    model->pool.stride = pool_size;

    // Calculate intermediate dimensions
    int conv_h = in_h - kernel_size + 1;
    int conv_w = in_w - kernel_size + 1;
    int pool_h = conv_h / pool_size;
    int pool_w = conv_w / pool_size;

    // Allocate persistent feature map buffers
    model->conv_out = m_new(float, out_channels * conv_h * conv_w);
    model->pool_out = m_new(float, out_channels * pool_h * pool_w);

    // 3. Setup Flatten -> Dense Layer dimensions
    int flattened_size = out_channels * pool_h * pool_w;

    // Prepend flattened dimension to the MLP layer configuration
    int *full_dense_sizes = m_new(int, num_dense_layers + 1);
    full_dense_sizes[0] = flattened_size;
    memcpy(&full_dense_sizes[1], dense_layers, num_dense_layers * sizeof(int));

    // Initialize dense classifier (classification mode: is_regression = 0)
    mlp_init(&model->dense, full_dense_sizes, num_dense_layers + 1, 0);
    m_del(int, full_dense_sizes, num_dense_layers + 1);
}

void cnn_free(CNNModel *model) {
    int w_size = model->conv.out_channels * model->conv.in_channels * 
                 model->conv.kernel_size * model->conv.kernel_size;
    
    if (model->conv.weights) m_del(float, model->conv.weights, w_size);
    if (model->conv.biases) m_del(float, model->conv.biases, model->conv.out_channels);

    int conv_h = model->in_h - model->conv.kernel_size + 1;
    int conv_w = model->in_w - model->conv.kernel_size + 1;
    int pool_h = conv_h / model->pool.pool_size;
    int pool_w = conv_w / model->pool.pool_size;

    if (model->conv_out) m_del(float, model->conv_out, model->conv.out_channels * conv_h * conv_w);
    if (model->pool_out) m_del(float, model->pool_out, model->conv.out_channels * pool_h * pool_w);

    mlp_free(&model->dense);
}

// Forward Conv2D pass + ReLU activation
static void conv2d_forward(const float *in, float *out, int in_h, int in_w, const Conv2DLayer *layer) {
    int out_h = in_h - layer->kernel_size + 1;
    int out_w = in_w - layer->kernel_size + 1;

    for (int oc = 0; oc < layer->out_channels; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = layer->biases[oc];
                for (int ic = 0; ic < layer->in_channels; ic++) {
                    for (int kh = 0; kh < layer->kernel_size; kh++) {
                        for (int kw = 0; kw < layer->kernel_size; kw++) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            int in_idx = (ic * in_h + ih) * in_w + iw;
                            int w_idx = ((oc * layer->in_channels + ic) * layer->kernel_size + kh) * layer->kernel_size + kw;
                            sum += in[in_idx] * layer->weights[w_idx];
                        }
                    }
                }
                out[(oc * out_h + oh) * out_w + ow] = (sum > 0.0f) ? sum : 0.0f; // ReLU
            }
        }
    }
}

// Forward MaxPool2D pass
static void maxpool2d_forward(const float *in, float *out, int in_c, int in_h, int in_w, const MaxPool2DLayer *pool) {
    int out_h = in_h / pool->pool_size;
    int out_w = in_w / pool->pool_size;

    for (int c = 0; c < in_c; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float max_val = -1e9f;
                for (int ph = 0; ph < pool->pool_size; ph++) {
                    for (int pw = 0; pw < pool->pool_size; pw++) {
                        int ih = oh * pool->stride + ph;
                        int iw = ow * pool->stride + pw;
                        float val = in[(c * in_h + ih) * in_w + iw];
                        if (val > max_val) max_val = val;
                    }
                }
                out[(c * out_h + oh) * out_w + ow] = max_val;
            }
        }
    }
}

int cnn_predict(CNNModel *model, const float *image, float *probs) {
    int conv_h = model->in_h - model->conv.kernel_size + 1;
    int conv_w = model->in_w - model->conv.kernel_size + 1;

    // 1. Conv2D -> ReLU
    conv2d_forward(image, model->conv_out, model->in_h, model->in_w, &model->conv);

    // 2. MaxPool2D
    maxpool2d_forward(model->conv_out, model->pool_out, model->conv.out_channels, conv_h, conv_w, &model->pool);

    // 3. Dense Classifier (using pool_out as flattened vector input)
    return mlp_predict(&model->dense, model->pool_out, probs);
}

int cnn_load(CNNModel *model, mp_obj_t file_obj) {
    int err = 0;
    
    // 1. Read Conv2D Weights & Biases
    int w_size = model->conv.out_channels * model->conv.in_channels * 
                 model->conv.kernel_size * model->conv.kernel_size;
    
    mp_stream_read_exactly(file_obj, model->conv.weights, w_size * sizeof(float), &err);
    mp_stream_read_exactly(file_obj, model->conv.biases, model->conv.out_channels * sizeof(float), &err);

    // 2. Read Dense Layer Weights & Biases
    int num_weight_matrices = model->dense.num_layers - 1;
    for (int l = 0; l < num_weight_matrices; l++) {
        int in_dim = model->dense.layer_sizes[l];
        int out_dim = model->dense.layer_sizes[l + 1];

        mp_stream_read_exactly(file_obj, model->dense.weights[l], in_dim * out_dim * sizeof(float), &err);
        mp_stream_read_exactly(file_obj, model->dense.biases[l], out_dim * sizeof(float), &err);
    }

    return err;
}
