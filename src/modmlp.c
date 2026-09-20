#include "py/runtime.h"
#include "py/objarray.h"
#include "mlp.h"
#include "py/mperrno.h"
// ulab headers
#include "../../micropython-ulab/code/ulab.h"
#include "../../micropython-ulab/code/ndarray.h"

extern const mp_obj_type_t ulab_ndarray_type;

#ifndef STATIC
#define STATIC static
#endif

typedef struct _mlp_obj_t {
    mp_obj_base_t base;
    MLPModel model;
    bool is_initialized;
} mlp_obj_t;

const mp_obj_type_t mlp_type;

// Helper to extract C float buffer from ulab ndarray
static float *get_ulab_float_ptr(mp_obj_t obj, size_t *len) {
    // Check against ulab_ndarray_type instead of ndarray_type
    if (!mp_obj_is_type(obj, &ulab_ndarray_type)) {
        mp_raise_TypeError(MP_ERROR_TEXT("expected ulab.ndarray"));
    }

    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(obj);

    // Verify element type is float
    if (ndarray->dtype != NDARRAY_FLOAT) {
        mp_raise_TypeError(MP_ERROR_TEXT("ndarray must be of float dtype"));
    }

    if (len != NULL) {
        *len = ndarray->len;
    }

    return (float *)ndarray->array;
}

// Helper to create a 1D or 2D ulab ndarray filled with zeros
static ndarray_obj_t* create_ulab_ndarray(size_t shape0, size_t shape1) {
    size_t shape[ULAB_MAX_DIMS] = {0};
    size_t ndim = 1;

    if (shape1 > 1) {
        shape[0] = shape0;
        shape[1] = shape1;
        ndim = 2;
    } else {
        shape[0] = shape0;
    }

    ndarray_obj_t *out = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
    return out;
}

// Constructor: mlp.MLP([inputs, hidden, output], is_regression=False)
STATIC mp_obj_t mlp_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 2, false);

    mlp_obj_t *self = m_new_obj(mlp_obj_t);
    self->base.type = type;
    self->is_initialized = false;

    if (n_args > 0 && mp_obj_is_type(args[0], &mp_type_list)) {
        mp_obj_list_t *list = MP_OBJ_TO_PTR(args[0]);
        int num_layers = list->len;
        int *layer_sizes = m_new(int, num_layers);

        for (size_t i = 0; i < num_layers; i++) {
            layer_sizes[i] = mp_obj_get_int(list->items[i]);
        }

        int is_regression = (n_args > 1) ? mp_obj_is_true(args[1]) : 0;
        mlp_init(&self->model, layer_sizes, num_layers, is_regression);
        m_del(int, layer_sizes, num_layers);
        self->is_initialized = true;
    }

    return MP_OBJ_FROM_PTR(self);
}

// model.fit(X_ulab, y_ulab, epochs=100, lr=0.01, momentum=0.9)
STATIC mp_obj_t mlp_fit_obj(size_t n_args, const mp_obj_t *args) {
    mlp_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    if (!self->is_initialized) mp_raise_ValueError(MP_ERROR_TEXT("Model uninitialized"));

    ndarray_obj_t *X_nd = MP_OBJ_TO_PTR(args[1]);
    float *X_ptr = get_ulab_float_ptr(args[1], 0);
    float *y_ptr = get_ulab_float_ptr(args[2], 0);

    int n_samples = (X_nd->ndim == 1) ? 1 : X_nd->shape[0];
    int epochs = (n_args > 3) ? mp_obj_get_int(args[3]) : 100;
    float lr = (n_args > 4) ? mp_obj_get_float(args[4]) : 0.01f;
    float momentum = (n_args > 5) ? mp_obj_get_float(args[5]) : 0.9f;

    mlp_fit(&self->model, X_ptr, y_ptr, n_samples, epochs, lr, momentum);
    return mp_const_none;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(mlp_fit_obj_f, 3, 6, mlp_fit_obj);

// model.predict(X_ulab)
// Handles single 1D vector or 2D batch of samples. Returns predicted class indices or output probabilities.
STATIC mp_obj_t mlp_predict_obj(mp_obj_t self_in, mp_obj_t x_in) {
    mlp_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (!self->is_initialized) mp_raise_ValueError(MP_ERROR_TEXT("Model uninitialized"));

    ndarray_obj_t *x_nd = MP_OBJ_TO_PTR(x_in);
    float *x_ptr = get_ulab_float_ptr(x_in, 0);

    int in_dim = self->model.layer_sizes[0];
    int out_dim = self->model.layer_sizes[self->model.num_layers - 1];

    if (x_nd->ndim == 1) {
        // Single sample prediction
        if (x_nd->len != in_dim) {
            mp_raise_ValueError(MP_ERROR_TEXT("Input size does not match network input layer size"));
        }

        ndarray_obj_t *out_nd = create_ulab_ndarray(out_dim, 1);
        float *out_ptr = (float*)out_nd->array;

        mlp_predict(&self->model, x_ptr, out_ptr);
        return MP_OBJ_FROM_PTR(out_nd);
    } else if (x_nd->ndim == 2) {
        // Batch prediction
        int n_samples = x_nd->shape[0];
        int sample_dim = x_nd->shape[1];

        if (sample_dim != in_dim) {
            mp_raise_ValueError(MP_ERROR_TEXT("Feature dimension does not match network input layer size"));
        }

        ndarray_obj_t *out_nd = create_ulab_ndarray(n_samples, out_dim);
        float *out_ptr = (float*)out_nd->array;

        for (int i = 0; i < n_samples; i++) {
            mlp_predict(&self->model, &x_ptr[i * in_dim], &out_ptr[i * out_dim]);
        }
        return MP_OBJ_FROM_PTR(out_nd);
    } else {
        mp_raise_ValueError(MP_ERROR_TEXT("Input array must be 1D or 2D"));
    }
}
STATIC MP_DEFINE_CONST_FUN_OBJ_2(mlp_predict_obj_f, mlp_predict_obj);

// model.predict_reg(X_ulab)
// Regression output evaluation over 1D vector or 2D batch.
STATIC mp_obj_t mlp_predict_reg_obj(mp_obj_t self_in, mp_obj_t x_in) {
    mlp_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (!self->is_initialized) mp_raise_ValueError(MP_ERROR_TEXT("Model uninitialized"));

    ndarray_obj_t *x_nd = MP_OBJ_TO_PTR(x_in);
    float *x_ptr = get_ulab_float_ptr(x_in, 0);

    int in_dim = self->model.layer_sizes[0];
    int out_dim = self->model.layer_sizes[self->model.num_layers - 1];

    if (x_nd->ndim == 1) {
        if (x_nd->len != in_dim) {
            mp_raise_ValueError(MP_ERROR_TEXT("Input size does not match network input layer size"));
        }

        ndarray_obj_t *out_nd = create_ulab_ndarray(out_dim, 1);
        float *out_ptr = (float*)out_nd->array;

        mlp_predict_reg(&self->model, x_ptr, out_ptr);
        return MP_OBJ_FROM_PTR(out_nd);
    } else if (x_nd->ndim == 2) {
        int n_samples = x_nd->shape[0];
        int sample_dim = x_nd->shape[1];

        if (sample_dim != in_dim) {
            mp_raise_ValueError(MP_ERROR_TEXT("Feature dimension does not match network input layer size"));
        }

        ndarray_obj_t *out_nd = create_ulab_ndarray(n_samples, out_dim);
        float *out_ptr = (float*)out_nd->array;

        for (int i = 0; i < n_samples; i++) {
            mlp_predict_reg(&self->model, &x_ptr[i * in_dim], &out_ptr[i * out_dim]);
        }
        return MP_OBJ_FROM_PTR(out_nd);
    } else {
        mp_raise_ValueError(MP_ERROR_TEXT("Input array must be 1D or 2D"));
    }
}
STATIC MP_DEFINE_CONST_FUN_OBJ_2(mlp_predict_reg_obj_f, mlp_predict_reg_obj);

// model.save("model.bin")
STATIC mp_obj_t mlp_save_obj(mp_obj_t self_in, mp_obj_t path_in) {
    mlp_obj_t *self = MP_OBJ_TO_PTR(self_in);
    const char *path = mp_obj_str_get_str(path_in);

    if (!mlp_save_model(&self->model, path)) {
        mp_raise_OSError(MP_EIO);
    }
    return mp_const_none;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_2(mlp_save_obj_f, mlp_save_obj);

// model.load("model.bin")
STATIC mp_obj_t mlp_load_obj(mp_obj_t self_in, mp_obj_t path_in) {
    mlp_obj_t *self = MP_OBJ_TO_PTR(self_in);
    const char *path = mp_obj_str_get_str(path_in);

    if (self->is_initialized) {
        mlp_free(&self->model);
        self->is_initialized = false;
    }

    if (!mlp_load_model(&self->model, path)) {
        mp_raise_OSError(MP_ENOENT);
    }
    self->is_initialized = true;
    return mp_const_none;
}
STATIC MP_DEFINE_CONST_FUN_OBJ_2(mlp_load_obj_f, mlp_load_obj);

// Module method table & types
STATIC const mp_rom_map_elem_t mlp_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_fit), MP_ROM_PTR(&mlp_fit_obj_f) },
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&mlp_predict_obj_f) },
    { MP_ROM_QSTR(MP_QSTR_predict_reg), MP_ROM_PTR(&mlp_predict_reg_obj_f) },
    { MP_ROM_QSTR(MP_QSTR_save), MP_ROM_PTR(&mlp_save_obj_f) },
    { MP_ROM_QSTR(MP_QSTR_load), MP_ROM_PTR(&mlp_load_obj_f) },
};
STATIC MP_DEFINE_CONST_DICT(mlp_locals_dict, mlp_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    mlp_type,
    MP_QSTR_MLP,
    MP_TYPE_FLAG_NONE,
    make_new, mlp_make_new,
    locals_dict, &mlp_locals_dict
);

STATIC const mp_rom_map_elem_t mp_module_mlp_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_mlp) },
    { MP_ROM_QSTR(MP_QSTR_MLP), MP_ROM_PTR(&mlp_type) },
};
STATIC MP_DEFINE_CONST_DICT(mp_module_mlp_globals, mp_module_mlp_globals_table);

const mp_obj_module_t mp_module_mlp = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&mp_module_mlp_globals,
};

MP_REGISTER_MODULE(MP_QSTR_mlp, mp_module_mlp);
