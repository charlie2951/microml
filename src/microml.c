#include <string.h>
#include "py/runtime.h"
#include "py/objarray.h"
#include "py/stream.h"
#include "py/builtin.h"
#include "microml.h"
#include "mlp.h"
#include "cnn.h"

#ifndef STATIC
#define STATIC static
#endif

// ==========================================
// HELPER: Convert flexible targets (ulab / array / list) to C int array
// ==========================================
static int* extract_int_targets(mp_obj_t y_in, size_t *out_n_samples) {
    mp_buffer_info_t buf_y;
    
    if (mp_get_buffer(y_in, &buf_y, MP_BUFFER_READ)) {
        // Handle buffer protocol (ulab ndarray, bytearray, array.array)
        size_t elem_size = 1;
        if (buf_y.typecode == 'i' || buf_y.typecode == 'I' || buf_y.typecode == 'f' || buf_y.typecode == 'l' || buf_y.typecode == 'L') {
            elem_size = 4;
        } else if (buf_y.typecode == 'h' || buf_y.typecode == 'H') {
            elem_size = 2;
        } else if (buf_y.typecode == 'd') {
            elem_size = 8;
        } else {
            elem_size = 1; // 'b', 'B', uint8_t, int8_t
        }

        *out_n_samples = buf_y.len / elem_size;
        int *y_targets = m_new(int, *out_n_samples);

        for (size_t i = 0; i < *out_n_samples; i++) {
            if (buf_y.typecode == 'f') {
                y_targets[i] = (int)(((float *)buf_y.buf)[i]);
            } else if (buf_y.typecode == 'd') {
                y_targets[i] = (int)(((double *)buf_y.buf)[i]);
            } else if (elem_size == 4) {
                y_targets[i] = ((int32_t *)buf_y.buf)[i];
            } else if (elem_size == 2) {
                y_targets[i] = ((int16_t *)buf_y.buf)[i];
            } else {
                y_targets[i] = ((uint8_t *)buf_y.buf)[i];
            }
        }
        return y_targets;
    } else if (mp_obj_is_type(y_in, &mp_type_list) || mp_obj_is_type(y_in, &mp_type_tuple)) {
        // Fallback for standard Python lists/tuples
        size_t len;
        mp_obj_t *items;
        mp_obj_get_array(y_in, &len, &items);
        *out_n_samples = len;
        int *y_targets = m_new(int, len);
        for (size_t i = 0; i < len; i++) {
            y_targets[i] = mp_obj_get_int(items[i]);
        }
        return y_targets;
    } else {
        mp_raise_TypeError(MP_ERROR_TEXT("Expected ulab ndarray, array, or list for y targets"));
        return NULL;
    }
}
//===========================================
// CNN WRAPPER
//============================================

typedef struct {
    mp_obj_base_t base;
    CNNModel model;
} mp_obj_cnn_t;

const mp_obj_type_t microml_cnn_type;

// Constructor: microml.CNN(image_shape, out_channels, kernel_size, pool_size, dense_layers)
// Example: microml.CNN((28, 28, 1), out_channels=4, kernel_size=3, pool_size=2, dense_layers=[16, 10])
static mp_obj_t cnn_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 5, 5, false);

    // Parse image_shape tuple (Height, Width, Channels)
    size_t shape_len;
    mp_obj_t *shape_items;
    mp_obj_get_array(args[0], &shape_len, &shape_items);
    
    int in_h = mp_obj_get_int(shape_items[0]);
    int in_w = mp_obj_get_int(shape_items[1]);
    int in_c = (shape_len >= 3) ? mp_obj_get_int(shape_items[2]) : 1;

    int out_channels = mp_obj_get_int(args[1]);
    int kernel_size = mp_obj_get_int(args[2]);
    int pool_size = mp_obj_get_int(args[3]);

    // Parse dense_layers array
    size_t num_dense;
    mp_obj_t *dense_items;
    mp_obj_get_array(args[4], &num_dense, &dense_items);

    int *dense_layers = m_new(int, num_dense);
    for (size_t i = 0; i < num_dense; i++) {
        dense_layers[i] = mp_obj_get_int(dense_items[i]);
    }

    mp_obj_cnn_t *self = m_new_obj(mp_obj_cnn_t);
    self->base.type = &microml_cnn_type;

    cnn_init(&self->model, in_h, in_w, in_c, out_channels, kernel_size, pool_size, dense_layers, (int)num_dense);
    m_del(int, dense_layers, num_dense);

    return MP_OBJ_FROM_PTR(self);
}

// Method: cnn.predict(image_buffer)
static mp_obj_t cnn_predict_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_cnn_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    int out_dim = self->model.dense.layer_sizes[self->model.dense.num_layers - 1];
    float *probs = m_new(float, out_dim);

    int pred_class = cnn_predict(&self->model, (float *)buf_x.buf, probs);
    m_del(float, probs, out_dim);

    return mp_obj_new_int(pred_class);
}
static MP_DEFINE_CONST_FUN_OBJ_2(cnn_predict_obj, cnn_predict_py);

// Method: cnn.load(filename)
static mp_obj_t cnn_load_py(mp_obj_t self_in, mp_obj_t filename_in) {
    mp_obj_cnn_t *self = MP_OBJ_TO_PTR(self_in);

    mp_obj_t open_args[2] = { filename_in, MP_OBJ_NEW_QSTR(MP_QSTR_rb) };
    mp_obj_t file = mp_builtin_open(2, open_args, (mp_map_t *)&mp_const_empty_map);

    cnn_load(&self->model, file);

    mp_stream_close(file);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(cnn_load_obj, cnn_load_py);

static const mp_rom_map_elem_t cnn_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&cnn_predict_obj) },
    { MP_ROM_QSTR(MP_QSTR_load), MP_ROM_PTR(&cnn_load_obj) },
};
static MP_DEFINE_CONST_DICT(cnn_locals_dict, cnn_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    microml_cnn_type,
    MP_QSTR_CNN,
    MP_TYPE_FLAG_NONE,
    make_new, cnn_make_new,
    locals_dict, &cnn_locals_dict
);

// ==========================================
// DEEP MULTI-LAYER PERCEPTRON (MLP) WRAPPER
// ==========================================
typedef struct {
    mp_obj_base_t base;
    MLPModel model;
} mp_obj_mlp_t;

const mp_obj_type_t microml_mlp_type;

// Constructor: microml.MLP(layer_sizes, is_regression=False)
static mp_obj_t mlp_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 1, 2, false);

    size_t num_layers;
    mp_obj_t *items;
    mp_obj_get_array(args[0], &num_layers, &items);

    if (num_layers < 2) {
        mp_raise_ValueError(MP_ERROR_TEXT("MLP requires at least 2 layer dimensions (input & output)"));
    }

    int *layer_sizes = m_new(int, num_layers);
    for (size_t i = 0; i < num_layers; i++) {
        layer_sizes[i] = mp_obj_get_int(items[i]);
    }

    int is_regression = (n_args >= 2) ? mp_obj_is_true(args[1]) : 0;

    mp_obj_mlp_t *self = m_new_obj(mp_obj_mlp_t);
    self->base.type = &microml_mlp_type;

    mlp_init(&self->model, layer_sizes, (int)num_layers, is_regression);

    m_del(int, layer_sizes, num_layers);
    return MP_OBJ_FROM_PTR(self);
}

// Method: mlp.fit(X, y, epochs=100, lr=0.01, momentum=0.9)
static mp_obj_t mlp_fit_py(size_t n_args, const mp_obj_t *args) {
    mp_obj_mlp_t *self = MP_OBJ_TO_PTR(args[0]);

    mp_buffer_info_t buf_x, buf_y;
    mp_get_buffer_raise(args[1], &buf_x, MP_BUFFER_READ);
    mp_get_buffer_raise(args[2], &buf_y, MP_BUFFER_READ);

    int epochs = (n_args >= 4) ? mp_obj_get_int(args[3]) : 100;
    float lr = (n_args >= 5) ? (float)mp_obj_get_float(args[4]) : 0.01f;
    float momentum = (n_args >= 6) ? (float)mp_obj_get_float(args[5]) : 0.9f;

    int input_dim = self->model.layer_sizes[0];
    int out_dim = self->model.layer_sizes[self->model.num_layers - 1];

    int n_samples;
    if (self->model.is_regression) {
        n_samples = buf_y.len / (sizeof(float) * out_dim);
    } else {
        n_samples = buf_x.len / (sizeof(float) * input_dim);
    }

    mlp_fit(&self->model, (float *)buf_x.buf, (float *)buf_y.buf, n_samples, epochs, lr, momentum);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(mlp_fit_obj, 3, 6, mlp_fit_py);

// Method: mlp.predict(X_sample)
static mp_obj_t mlp_predict_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_mlp_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    int out_dim = self->model.layer_sizes[self->model.num_layers - 1];

    if (self->model.is_regression) {
        float *out_pred = m_new(float, out_dim);
        mlp_predict_reg(&self->model, (float *)buf_x.buf, out_pred);

        mp_obj_t pred_list = mp_obj_new_list(0, NULL);
        for (int i = 0; i < out_dim; i++) {
            mp_obj_list_append(pred_list, mp_obj_new_float((mp_float_t)out_pred[i]));
        }
        m_del(float, out_pred, out_dim);
        return pred_list;
    } else {
        float *probs = m_new(float, out_dim);
        int pred_class = mlp_predict(&self->model, (float *)buf_x.buf, probs);
        m_del(float, probs, out_dim);
        return mp_obj_new_int(pred_class);
    }
}
static MP_DEFINE_CONST_FUN_OBJ_2(mlp_predict_obj, mlp_predict_py);

// Method: mlp.predict_proba(X_sample)
static mp_obj_t mlp_predict_proba_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_mlp_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    int out_dim = self->model.layer_sizes[self->model.num_layers - 1];
    float *probs = m_new(float, out_dim);
    mlp_predict(&self->model, (float *)buf_x.buf, probs);

    mp_obj_t prob_list = mp_obj_new_list(0, NULL);
    for (int i = 0; i < out_dim; i++) {
        mp_obj_list_append(prob_list, mp_obj_new_float((mp_float_t)probs[i]));
    }
    m_del(float, probs, out_dim);
    return prob_list;
}
static MP_DEFINE_CONST_FUN_OBJ_2(mlp_predict_proba_obj, mlp_predict_proba_py);

// Method: mlp.save(filename)
static mp_obj_t mlp_save_py(mp_obj_t self_in, mp_obj_t filename_in) {
    mp_obj_mlp_t *self = MP_OBJ_TO_PTR(self_in);

    // Open file in write-binary mode using MicroPython VFS
    mp_obj_t open_args[2] = { filename_in, MP_OBJ_NEW_QSTR(MP_QSTR_wb) };
    mp_obj_t file = mp_builtin_open(2, open_args, (mp_map_t *)&mp_const_empty_map);

    int err = 0;

    // 1. Write Header Metadata: num_layers, is_regression
    int header[2] = { self->model.num_layers, self->model.is_regression };
    mp_stream_write_exactly(file, header, sizeof(header), &err);

    // 2. Write layer_sizes array
    mp_stream_write_exactly(file, self->model.layer_sizes, self->model.num_layers * sizeof(int), &err);

    // 3. Write Weights and Biases layer by layer
    int num_weight_matrices = self->model.num_layers - 1;
    for (int l = 0; l < num_weight_matrices; l++) {
        int in_dim = self->model.layer_sizes[l];
        int out_dim = self->model.layer_sizes[l + 1];

        // Write weight matrix [in_dim * out_dim]
        mp_stream_write_exactly(file, self->model.weights[l], in_dim * out_dim * sizeof(float), &err);
        
        // Write bias array [out_dim]
        mp_stream_write_exactly(file, self->model.biases[l], out_dim * sizeof(float), &err);
    }

    mp_stream_close(file);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(mlp_save_obj, mlp_save_py);

// Method: mlp.load(filename)
static mp_obj_t mlp_load_py(mp_obj_t self_in, mp_obj_t filename_in) {
    mp_obj_mlp_t *self = MP_OBJ_TO_PTR(self_in);

    // Open file in read-binary mode using MicroPython VFS
    mp_obj_t open_args[2] = { filename_in, MP_OBJ_NEW_QSTR(MP_QSTR_rb) };
    mp_obj_t file = mp_builtin_open(2, open_args, (mp_map_t *)&mp_const_empty_map);

    int err = 0;

    // 1. Read Header Metadata
    int header[2];
    mp_stream_read_exactly(file, header, sizeof(header), &err);

    int num_layers = header[0];
    int is_regression = header[1];

    // 2. Read layer dimensions
    int *layer_sizes = m_new(int, num_layers);
    mp_stream_read_exactly(file, layer_sizes, num_layers * sizeof(int), &err);

    // 3. Free old allocations if model is already initialized
    if (self->model.layer_sizes != NULL) {
        mlp_free(&self->model);
    }

    // 4. Re-allocate memory structures for network topology
    mlp_init(&self->model, layer_sizes, num_layers, is_regression);
    m_del(int, layer_sizes, num_layers);

    // 5. Read binary Weights and Biases back into allocated memory buffers
    int num_weight_matrices = self->model.num_layers - 1;
    for (int l = 0; l < num_weight_matrices; l++) {
        int in_dim = self->model.layer_sizes[l];
        int out_dim = self->model.layer_sizes[l + 1];

        mp_stream_read_exactly(file, self->model.weights[l], in_dim * out_dim * sizeof(float), &err);
        mp_stream_read_exactly(file, self->model.biases[l], out_dim * sizeof(float), &err);
    }

    mp_stream_close(file);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(mlp_load_obj, mlp_load_py);

static const mp_rom_map_elem_t mlp_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_fit), MP_ROM_PTR(&mlp_fit_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&mlp_predict_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict_proba), MP_ROM_PTR(&mlp_predict_proba_obj) },
    { MP_ROM_QSTR(MP_QSTR_save), MP_ROM_PTR(&mlp_save_obj) },
    { MP_ROM_QSTR(MP_QSTR_load), MP_ROM_PTR(&mlp_load_obj) },
};
static MP_DEFINE_CONST_DICT(mlp_locals_dict, mlp_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    microml_mlp_type,
    MP_QSTR_MLP,
    MP_TYPE_FLAG_NONE,
    make_new, mlp_make_new,
    locals_dict, &mlp_locals_dict
);

// ==========================================
// KNN WRAPPER
// ==========================================
typedef struct {
    mp_obj_base_t base;
    KNNModel model;
} mp_obj_knn_t;

const mp_obj_type_t microml_knn_type;

static mp_obj_t knn_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 1, false);
    int k = (n_args == 1) ? mp_obj_get_int(args[0]) : 3;

    mp_obj_knn_t *self = m_new_obj(mp_obj_knn_t);
    self->base.type = &microml_knn_type;
    knn_init(&self->model, k);
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t knn_fit_py(size_t n_args, const mp_obj_t *args) {
    mp_obj_knn_t *self = MP_OBJ_TO_PTR(args[0]);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(args[1], &buf_x, MP_BUFFER_READ);

    size_t n_samples;
    int *y_targets = extract_int_targets(args[2], &n_samples);

    int n_features = mp_obj_get_int(args[3]);
    int n_classes = (n_args >= 5) ? mp_obj_get_int(args[4]) : 2;

    knn_fit(&self->model, (float *)buf_x.buf, y_targets, (int)n_samples, n_features, n_classes);
    
    m_del(int, y_targets, n_samples);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(knn_fit_obj, 4, 5, knn_fit_py);

static mp_obj_t knn_predict_proba_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_knn_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    float *probs = m_new(float, self->model.n_classes);
    int pred = knn_predict_proba(&self->model, (float *)buf_x.buf, probs);

    mp_obj_t tuple[2];
    tuple[0] = mp_obj_new_int(pred);

    mp_obj_t prob_list = mp_obj_new_list(0, NULL);
    for (int i = 0; i < self->model.n_classes; i++) {
        mp_obj_list_append(prob_list, mp_obj_new_float((mp_float_t)probs[i]));
    }
    tuple[1] = prob_list;

    m_del(float, probs, self->model.n_classes);
    return mp_obj_new_tuple(2, tuple);
}
static MP_DEFINE_CONST_FUN_OBJ_2(knn_predict_proba_obj, knn_predict_proba_py);

static const mp_rom_map_elem_t knn_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_fit), MP_ROM_PTR(&knn_fit_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict_proba), MP_ROM_PTR(&knn_predict_proba_obj) },
};
static MP_DEFINE_CONST_DICT(knn_locals_dict, knn_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    microml_knn_type,
    MP_QSTR_KNN,
    MP_TYPE_FLAG_NONE,
    make_new, knn_make_new,
    locals_dict, &knn_locals_dict
);

// ==========================================
// DECISION TREE WRAPPER
// ==========================================
typedef struct {
    mp_obj_base_t base;
    DTModel model;
} mp_obj_dt_t;

const mp_obj_type_t microml_dt_type;

static mp_obj_t dt_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 1, false);
    int max_depth = (n_args == 1) ? mp_obj_get_int(args[0]) : 5;

    mp_obj_dt_t *self = m_new_obj(mp_obj_dt_t);
    self->base.type = &microml_dt_type;
    dt_init(&self->model, max_depth);
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t dt_fit_py(size_t n_args, const mp_obj_t *args) {
    mp_obj_dt_t *self = MP_OBJ_TO_PTR(args[0]);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(args[1], &buf_x, MP_BUFFER_READ);

    size_t n_samples;
    int *y_targets = extract_int_targets(args[2], &n_samples);

    int n_features = mp_obj_get_int(args[3]);
    int n_classes = (n_args >= 5) ? mp_obj_get_int(args[4]) : 2;

    dt_fit(&self->model, (float *)buf_x.buf, y_targets, (int)n_samples, n_features, n_classes);

    m_del(int, y_targets, n_samples);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(dt_fit_obj, 4, 5, dt_fit_py);

static mp_obj_t dt_predict_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_dt_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    float confidence = 0.0f;
    int pred = dt_predict(&self->model, (float *)buf_x.buf, &confidence);

    mp_obj_t tuple[2] = { mp_obj_new_int(pred), mp_obj_new_float((mp_float_t)confidence) };
    return mp_obj_new_tuple(2, tuple);
}
static MP_DEFINE_CONST_FUN_OBJ_2(dt_predict_obj, dt_predict_py);

static mp_obj_t dt_save_py(mp_obj_t self_in, mp_obj_t filename_in) {
    mp_obj_dt_t *self = MP_OBJ_TO_PTR(self_in);

    mp_obj_t open_args[2] = { filename_in, MP_OBJ_NEW_QSTR(MP_QSTR_wb) };
    mp_obj_t file = mp_builtin_open(2, open_args, (mp_map_t *)&mp_const_empty_map);

    int header[3] = { self->model.node_count, self->model.n_features, self->model.n_classes };
    int err = 0;
    mp_stream_write_exactly(file, header, sizeof(header), &err);
    mp_stream_write_exactly(file, self->model.nodes, self->model.node_count * sizeof(TreeNode), &err);

    mp_stream_close(file);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(dt_save_obj, dt_save_py);

static mp_obj_t dt_load_py(mp_obj_t self_in, mp_obj_t filename_in) {
    mp_obj_dt_t *self = MP_OBJ_TO_PTR(self_in);

    mp_obj_t open_args[2] = { filename_in, MP_OBJ_NEW_QSTR(MP_QSTR_rb) };
    mp_obj_t file = mp_builtin_open(2, open_args, (mp_map_t *)&mp_const_empty_map);

    int header[3];
    int err = 0;
    mp_stream_read_exactly(file, header, sizeof(header), &err);

    if (self->model.nodes) {
        m_del(TreeNode, self->model.nodes, self->model.node_count);
    }

    self->model.node_count = header[0];
    self->model.n_features = header[1];
    self->model.n_classes = header[2];

    self->model.nodes = m_new(TreeNode, self->model.node_count);
    mp_stream_read_exactly(file, self->model.nodes, self->model.node_count * sizeof(TreeNode), &err);

    mp_stream_close(file);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(dt_load_obj, dt_load_py);

static const mp_rom_map_elem_t dt_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_fit), MP_ROM_PTR(&dt_fit_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&dt_predict_obj) },
    { MP_ROM_QSTR(MP_QSTR_save), MP_ROM_PTR(&dt_save_obj) },
    { MP_ROM_QSTR(MP_QSTR_load), MP_ROM_PTR(&dt_load_obj) },
};
static MP_DEFINE_CONST_DICT(dt_locals_dict, dt_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    microml_dt_type,
    MP_QSTR_DecisionTree,
    MP_TYPE_FLAG_NONE,
    make_new, dt_make_new,
    locals_dict, &dt_locals_dict
);

// ==========================================
// SVM WRAPPER (LINEAR & RBF)
// ==========================================
typedef struct {
    mp_obj_base_t base;
    SVMModel model;
} mp_obj_svm_t;

const mp_obj_type_t microml_svm_type;

static mp_obj_t svm_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 2, false);
    int kernel = (n_args >= 1) ? mp_obj_get_int(args[0]) : SVM_KERNEL_LINEAR;
    float gamma = (n_args >= 2) ? (float)mp_obj_get_float(args[1]) : 0.5f;

    mp_obj_svm_t *self = m_new_obj(mp_obj_svm_t);
    self->base.type = &microml_svm_type;
    svm_init(&self->model, kernel, gamma);
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t svm_fit_py(size_t n_args, const mp_obj_t *args) {
    mp_obj_svm_t *self = MP_OBJ_TO_PTR(args[0]);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(args[1], &buf_x, MP_BUFFER_READ);

    size_t n_samples;
    int *y_targets = extract_int_targets(args[2], &n_samples);

    int n_features = mp_obj_get_int(args[3]);
    int epochs = (n_args >= 5) ? mp_obj_get_int(args[4]) : 100;
    float lr = (n_args >= 6) ? (float)mp_obj_get_float(args[5]) : 0.01f;
    float C = (n_args >= 7) ? (float)mp_obj_get_float(args[6]) : 1.0f;

    svm_fit(&self->model, (float *)buf_x.buf, y_targets, (int)n_samples, n_features, epochs, lr, C);

    m_del(int, y_targets, n_samples);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(svm_fit_obj, 4, 7, svm_fit_py);

static mp_obj_t svm_predict_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_svm_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    int pred = svm_predict(&self->model, (float *)buf_x.buf);
    return mp_obj_new_int(pred);
}
static MP_DEFINE_CONST_FUN_OBJ_2(svm_predict_obj, svm_predict_py);

static const mp_rom_map_elem_t svm_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_fit), MP_ROM_PTR(&svm_fit_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&svm_predict_obj) },
};
static MP_DEFINE_CONST_DICT(svm_locals_dict, svm_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    microml_svm_type,
    MP_QSTR_SVM,
    MP_TYPE_FLAG_NONE,
    make_new, svm_make_new,
    locals_dict, &svm_locals_dict
);

// ==========================================
// MODULE REGISTRATION
// ==========================================
static const mp_rom_map_elem_t microml_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_microml) },
    { MP_ROM_QSTR(MP_QSTR_KNN), MP_ROM_PTR(&microml_knn_type) },
    { MP_ROM_QSTR(MP_QSTR_DecisionTree), MP_ROM_PTR(&microml_dt_type) },
    { MP_ROM_QSTR(MP_QSTR_SVM), MP_ROM_PTR(&microml_svm_type) },
    { MP_ROM_QSTR(MP_QSTR_MLP), MP_ROM_PTR(&microml_mlp_type) },
    { MP_ROM_QSTR(MP_QSTR_KERNEL_LINEAR), MP_ROM_INT(SVM_KERNEL_LINEAR) },
    { MP_ROM_QSTR(MP_QSTR_KERNEL_RBF), MP_ROM_INT(SVM_KERNEL_RBF) },
    { MP_ROM_QSTR(MP_QSTR_CNN), MP_ROM_PTR(&microml_cnn_type) },//cnn module registration
};
static MP_DEFINE_CONST_DICT(microml_module_globals, microml_module_globals_table);

const mp_obj_module_t microml_usercmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&microml_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR__microml, microml_usercmodule);