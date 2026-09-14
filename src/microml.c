#include <string.h>
#include "py/runtime.h"
#include "py/objarray.h"
#include "py/stream.h"
#include "py/builtin.h"
#include "microml.h"
#include "mlp.h"

// ==========================================
// MLP REGRESSOR WRAPPER
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
    mlp_init(&self->model, input_dim, hidden_dim, output_dim, 1);
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
// KNN REGRESSOR WRAPPER
// ==========================================
typedef struct {
    mp_obj_base_t base;
    KNNRegressorModel model;
} mp_obj_knn_reg_t;

const mp_obj_type_t microml_knn_reg_type;

static mp_obj_t knn_reg_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 1, false);
    int k = (n_args == 1) ? mp_obj_get_int(args[0]) : 3;

    mp_obj_knn_reg_t *self = m_new_obj(mp_obj_knn_reg_t);
    self->base.type = &microml_knn_reg_type;
    knn_reg_init(&self->model, k);
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t knn_reg_fit_py(mp_obj_t self_in, mp_obj_t x_in, mp_obj_t y_in, mp_obj_t n_feats_in) {
    mp_obj_knn_reg_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x, buf_y;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);
    mp_get_buffer_raise(y_in, &buf_y, MP_BUFFER_READ);

    int n_features = mp_obj_get_int(n_feats_in);
    int n_samples = buf_y.len / sizeof(float);

    knn_reg_fit(&self->model, (float *)buf_x.buf, (float *)buf_y.buf, n_samples, n_features);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_4(knn_reg_fit_obj, knn_reg_fit_py);

static mp_obj_t knn_reg_predict_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_knn_reg_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    float pred = knn_reg_predict(&self->model, (float *)buf_x.buf);
    return mp_obj_new_float((mp_float_t)pred);
}
static MP_DEFINE_CONST_FUN_OBJ_2(knn_reg_predict_obj, knn_reg_predict_py);

static const mp_rom_map_elem_t knn_reg_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_fit), MP_ROM_PTR(&knn_reg_fit_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&knn_reg_predict_obj) },
};
static MP_DEFINE_CONST_DICT(knn_reg_locals_dict, knn_reg_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    microml_knn_reg_type,
    MP_QSTR_KNNRegressor,
    MP_TYPE_FLAG_NONE,
    make_new, knn_reg_make_new,
    locals_dict, &knn_reg_locals_dict
);

// ==========================================
// DECISION TREE REGRESSOR WRAPPER
// ==========================================
typedef struct {
    mp_obj_base_t base;
    DTRegressorModel model;
} mp_obj_dt_reg_t;

const mp_obj_type_t microml_dt_reg_type;

static mp_obj_t dt_reg_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 1, false);
    int max_depth = (n_args == 1) ? mp_obj_get_int(args[0]) : 5;

    mp_obj_dt_reg_t *self = m_new_obj(mp_obj_dt_reg_t);
    self->base.type = &microml_dt_reg_type;
    dt_reg_init(&self->model, max_depth);
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t dt_reg_fit_py(mp_obj_t self_in, mp_obj_t x_in, mp_obj_t y_in, mp_obj_t n_feats_in) {
    mp_obj_dt_reg_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x, buf_y;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);
    mp_get_buffer_raise(y_in, &buf_y, MP_BUFFER_READ);

    int n_features = mp_obj_get_int(n_feats_in);
    int n_samples = buf_y.len / sizeof(float);

    dt_reg_fit(&self->model, (float *)buf_x.buf, (float *)buf_y.buf, n_samples, n_features);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_4(dt_reg_fit_obj, dt_reg_fit_py);

static mp_obj_t dt_reg_predict_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_dt_reg_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    float pred = dt_reg_predict(&self->model, (float *)buf_x.buf);
    return mp_obj_new_float((mp_float_t)pred);
}
static MP_DEFINE_CONST_FUN_OBJ_2(dt_reg_predict_obj, dt_reg_predict_py);

static const mp_rom_map_elem_t dt_reg_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_fit), MP_ROM_PTR(&dt_reg_fit_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&dt_reg_predict_obj) },
};
static MP_DEFINE_CONST_DICT(dt_reg_locals_dict, dt_reg_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    microml_dt_reg_type,
    MP_QSTR_DecisionTreeRegressor,
    MP_TYPE_FLAG_NONE,
    make_new, dt_reg_make_new,
    locals_dict, &dt_reg_locals_dict
);

// ==========================================
// SVR WRAPPER
// ==========================================
typedef struct {
    mp_obj_base_t base;
    SVRModel model;
} mp_obj_svr_t;

const mp_obj_type_t microml_svr_type;

static mp_obj_t svr_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_obj_svr_t *self = m_new_obj(mp_obj_svr_t);
    self->base.type = &microml_svr_type;
    svr_init(&self->model);
    return MP_OBJ_FROM_PTR(self);
}

static mp_obj_t svr_fit_py(size_t n_args, const mp_obj_t *args) {
    mp_obj_svr_t *self = MP_OBJ_TO_PTR(args[0]);
    mp_buffer_info_t buf_x, buf_y;
    mp_get_buffer_raise(args[1], &buf_x, MP_BUFFER_READ);
    mp_get_buffer_raise(args[2], &buf_y, MP_BUFFER_READ);

    int n_features = mp_obj_get_int(args[3]);
    int epochs = (n_args >= 5) ? mp_obj_get_int(args[4]) : 100;
    float lr = (n_args >= 6) ? (float)mp_obj_get_float(args[5]) : 0.01f;
    float epsilon = (n_args >= 7) ? (float)mp_obj_get_float(args[6]) : 0.1f;

    int n_samples = buf_y.len / sizeof(float);
    svr_fit(&self->model, (float *)buf_x.buf, (float *)buf_y.buf, n_samples, n_features, epochs, lr, epsilon);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(svr_fit_obj, 4, 7, svr_fit_py);

static mp_obj_t svr_predict_py(mp_obj_t self_in, mp_obj_t x_in) {
    mp_obj_svr_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t buf_x;
    mp_get_buffer_raise(x_in, &buf_x, MP_BUFFER_READ);

    float pred = svr_predict(&self->model, (float *)buf_x.buf);
    return mp_obj_new_float((mp_float_t)pred);
}
static MP_DEFINE_CONST_FUN_OBJ_2(svr_predict_obj, svr_predict_py);

static const mp_rom_map_elem_t svr_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_fit), MP_ROM_PTR(&svr_fit_obj) },
    { MP_ROM_QSTR(MP_QSTR_predict), MP_ROM_PTR(&svr_predict_obj) },
};
static MP_DEFINE_CONST_DICT(svr_locals_dict, svr_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    microml_svr_type,
    MP_QSTR_SVR,
    MP_TYPE_FLAG_NONE,
    make_new, svr_make_new,
    locals_dict, &svr_locals_dict
);

// ==========================================
// MODULE REGISTRATION
// ==========================================
static const mp_rom_map_elem_t microml_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_microml) },
    { MP_ROM_QSTR(MP_QSTR_MLPRegressor), MP_ROM_PTR(&microml_mlp_reg_type) },
    { MP_ROM_QSTR(MP_QSTR_KNNRegressor), MP_ROM_PTR(&microml_knn_reg_type) },
    { MP_ROM_QSTR(MP_QSTR_DecisionTreeRegressor), MP_ROM_PTR(&microml_dt_reg_type) },
    { MP_ROM_QSTR(MP_QSTR_SVR), MP_ROM_PTR(&microml_svr_type) },
};
static MP_DEFINE_CONST_DICT(microml_module_globals, microml_module_globals_table);

const mp_obj_module_t microml_usercmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&microml_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_microml, microml_usercmodule);
