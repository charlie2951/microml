#ifndef CNN_H
#define CNN_H

#include "py/runtime.h"
#include "mlp.h"

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    float *weights; // Size: out_channels * in_channels * kernel_size * kernel_size
    float *biases;  // Size: out_channels
} Conv2DLayer;

typedef struct {
    int pool_size;
    int stride;
} MaxPool2DLayer;

typedef struct {
    int in_h, in_w, in_c;    // Input image dimensions (Height, Width, Channels)
    
    Conv2DLayer conv;
    MaxPool2DLayer pool;
    MLPModel dense;          // Reuses your existing MLP structure for final layers

    // Scratchpad activation buffers to avoid heap fragmentation during inference
    float *conv_out;
    float *pool_out;
} CNNModel;

void cnn_init(CNNModel *model, int in_h, int in_w, int in_c, 
              int out_channels, int kernel_size, int pool_size,
              const int *dense_layers, int num_dense_layers);

void cnn_free(CNNModel *model);
int cnn_predict(CNNModel *model, const float *image, float *probs);
int cnn_load(CNNModel *model, mp_obj_t file_obj);

#endif // CNN_H
