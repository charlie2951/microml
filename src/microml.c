#include <string.h>
#include "py/runtime.h"
#include "py/objarray.h"
#include "py/stream.h"
#include "py/builtin.h"
#include "microml.h"
#include "mlp.h"

// ==========================================
// 1. MLP CLASSIFIER WRAPPER
// ==========================================
typedef struct {
    mp_obj_base_t base;
    MLPModel model;
} mp_obj_mlp_t;

const mp_obj_type_t microml_mlp_type;

static mp_obj_t mlp_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 3, 3, false);

    int input_dim = mp_obj_get_int(args[0]);
    int hidden_dim = mp_obj_get_int(args[1]);
    int output_dim = mp_obj_get_int(args[2]);

    mp_obj_mlp_t *self = m_new_obj(mp_obj_mlp_t);
    self->base.type = &microml_mlp_type;
    mlp_init(&self->model, input_dim, hidden_dim, output_dim, 0); // 0 = Classification
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t mlp_fit_py(size_t n_args, const mp_obj_t *args) {
    mp_obj_mlp_t *self = MP_OBJ_TO_PTR(args[0]);

    mp_buffer_info_t buf_x, buf_y;
    mp_get_buffer_raise(args[1], &buf_x, MP_BUFFER_READ);
    mp_get_buffer_raise(args[2], &buf_y, MP_BUFFER_READ);

    int epochs = (n_args >= 4) ? mp_obj_get_int(args[3]) : 100;
    float lr = (n_args >= 5) ? (float)mp_obj_get_float(args[4]) : 0.01f;
    float momentum = (n_args >= 6) ? (float)mp_obj_get_float(args[5]) : 0.9f;

    int n_samples = buf_y.len / sizeof(int);

    // Cast y to float array temporarily for uniform function call
    float *y_float = m_new(float, n_samples);
    int *y_int = (int *)buf_y.buf;
    for (int i = 0; i < n_samples; i++) y_float[i] = (float)y_int[i];

    mlp_fit(&self->model, (float *)buf_x.buf, y_float, n_samples, epochs, lr, momentum);

    m_del(float, y_float, n_samples);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(mlp_fit_obj, 3, 6, mlp_fit_py);

static mp_obj_t mlp_predict_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_mlp_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    float *probs = m_new(float, self->model.output_dim);
    int pred = mlp_predict(&self->model, (float *)buf_x.buf, probs);

    mp_obj_t prob_list = mp_obj_new_list(0, NULL);
    for (int i = 0; i < self->model.output_dim; i++) {
        mp_obj_list_append(prob_list, mp_obj_new_float((mp_float_t)probs[i]));
    }

    mp_obj_t tuple[2] = { mp_obj_new_int(pred), prob_list };

    m_del(float, probs, self->model.output_dim);
    return mp_obj_new_tuple(2, tuple);
}
static MP_DEFINE_CONST_FUN_OBJ_2(mlp_predict_obj, mlp_predict_py);

static const mp_rom_map_elem_t mlp_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_fit), MP_ROM_PTR(&mlp_fit_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&mlp_predict_obj) },
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
// 2. MLP REGRESSOR WRAPPER (NEW)
// ==========================================
typedef struct {
    mp_obj_base_t base;
    MLPModel model;
} mp_obj_mlp_reg_t;

const mp_obj_type_t microml_mlp_reg_type;

static mp_obj_t mlp_reg_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 2, 3, false);

    int input_dim = mp_obj_get_int(args[0]);
    int hidden_dim = mp_obj_get_int(args[1]);
    int output_dim = (n_args >= 3) ? mp_obj_get_int(args[2]) : 1;

    mp_obj_mlp_reg_t *self = m_new_obj(mp_obj_mlp_reg_t);
    self->base.type = &microml_mlp_reg_type;
    mlp_init(&self->model, input_dim, hidden_dim, output_dim, 1); // 1 = Regression
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t mlp_reg_fit_py(size_t n_args, const mp_obj_t *args) {
    mp_obj_mlp_reg_t *self = MP_OBJ_TO_PTR(args[0]);

    mp_buffer_info_t buf_x, buf_y;
    mp_get_buffer_raise(args[1], &buf_x, MP_BUFFER_READ);
    mp_get_buffer_raise(args[2], &buf_y, MP_BUFFER_READ);

    int epochs = (n_args >= 4) ? mp_obj_get_int(args[3]) : 100;
    float lr = (n_args >= 5) ? (float)mp_obj_get_float(args[4]) : 0.01f;
    float momentum = (n_args >= 6) ? (float)mp_obj_get_float(args[5]) : 0.9f;

    int n_samples = buf_y.len / (sizeof(float) * self->model.output_dim);

    mlp_fit(&self->model, (float *)buf_x.buf, (float *)buf_y.buf, n_samples, epochs, lr, momentum);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(mlp_reg_fit_obj, 3, 6, mlp_reg_fit_py);

static mp_obj_t mlp_reg_predict_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_mlp_reg_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    float *out = m_new(float, self->model.output_dim);
    mlp_predict_reg(&self->model, (float *)buf_x.buf, out);

    if (self->model.output_dim == 1) {
        float val = out[0];
        m_del(float, out, 1);
        return mp_obj_new_float((mp_float_t)val);
    }

    mp_obj_t list = mp_obj_new_list(0, NULL);
    for (int i = 0; i < self->model.output_dim; i++) {
        mp_obj_list_append(list, mp_obj_new_float((mp_float_t)out[i]));
    }
    m_del(float, out, self->model.output_dim);
    return list;
}
static MP_DEFINE_CONST_FUN_OBJ_2(mlp_reg_predict_obj, mlp_reg_predict_py);

static const mp_rom_map_elem_t mlp_reg_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_fit), MP_ROM_PTR(&mlp_reg_fit_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&mlp_reg_predict_obj) },
};
static MP_DEFINE_CONST_DICT(mlp_reg_locals_dict, mlp_reg_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    microml_mlp_reg_type,
    MP_QSTR_MLPRegressor,
    MP_TYPE_FLAG_NONE,
    make_new, mlp_reg_make_new,
    locals_dict, &mlp_reg_locals_dict
);

// ==========================================
// 3. KNN WRAPPER
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
    mp_buffer_info_t buf_x, buf_y;
    mp_get_buffer_raise(args[1], &buf_x, MP_BUFFER_READ);
    mp_get_buffer_raise(args[2], &buf_y, MP_BUFFER_READ);

    int n_features = mp_obj_get_int(args[3]);
    int n_classes = (n_args >= 5) ? mp_obj_get_int(args[4]) : 2;

    int n_samples = buf_y.len / sizeof(int);
    knn_fit(&self->model, (float *)buf_x.buf, (int *)buf_y.buf, n_samples, n_features, n_classes);
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
// 4. DECISION TREE WRAPPER
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
    mp_buffer_info_t buf_x, buf_y;
    mp_get_buffer_raise(args[1], &buf_x, MP_BUFFER_READ);
    mp_get_buffer_raise(args[2], &buf_y, MP_BUFFER_READ);

    int n_features = mp_obj_get_int(args[3]);
    int n_classes = (n_args >= 5) ? mp_obj_get_int(args[4]) : 2;

    int n_samples = buf_y.len / sizeof(int);
    dt_fit(&self->model, (float *)buf_x.buf, (int *)buf_y.buf, n_samples, n_features, n_classes);
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
// 5. SVM WRAPPER
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
    mp_buffer_info_t buf_x, buf_y;
    mp_get_buffer_raise(args[1], &buf_x, MP_BUFFER_READ);
    mp_get_buffer_raise(args[2], &buf_y, MP_BUFFER_READ);

    int n_features = mp_obj_get_int(args[3]);
    int epochs = (n_args >= 5) ? mp_obj_get_int(args[4]) : 100;
    float lr = (n_args >= 6) ? (float)mp_obj_get_float(args[5]) : 0.01f;
    float C = (n_args >= 7) ? (float)mp_obj_get_float(args[6]) : 1.0f;

    int n_samples = buf_y.len / sizeof(int);
    svm_fit(&self->model, (float *)buf_x.buf, (int *)buf_y.buf, n_samples, n_features, epochs, lr, C);
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
    { MP_ROM_QSTR(MP_QSTR_MLPRegressor), MP_ROM_PTR(&microml_mlp_reg_type) },
    { MP_ROM_QSTR(MP_QSTR_KERNEL_LINEAR), MP_ROM_INT(SVM_KERNEL_LINEAR) },
    { MP_ROM_QSTR(MP_QSTR_KERNEL_RBF), MP_ROM_INT(SVM_KERNEL_RBF) },
};
static MP_DEFINE_CONST_DICT(microml_module_globals, microml_module_globals_table);

const mp_obj_module_t microml_usercmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&microml_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_microml, microml_usercmodule);