//
// caffe_.cpp provides wrappers of the caffe::Solver class, caffe::Net class,
// caffe::Layer class and caffe::Blob class and some caffe::Caffe functions,
// so that one could easily use Caffe from matlab.
// Note that for matlab, we will simply use float as the data type.

// Internally, data is stored with dimensions reversed from Caffe's:
// e.g., if the Caffe blob axes are (num, channels, height, width),
// the matcaffe data is stored as (width, height, channels, num)
// where width is the fastest dimension.

#include <sstream>
#include <string>
#include <vector>

#include "mex.h"

#include "caffe/caffe.hpp"
#include "caffe/util/im2col.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;  // NOLINT(build/namespaces)

// Do CHECK and throw a Mex error if check fails
inline void mxCHECK(bool expr, const char* msg) {
  if (!expr) {
    mexErrMsgTxt(msg);
  }
}
inline void mxERROR(const char* msg) { mexErrMsgTxt(msg); }

// Check if a file exists and can be opened
void mxCHECK_FILE_EXIST(const char* file) {
  std::ifstream f(file);
  if (!f.good()) {
    f.close();
    std::string msg("Could not open file ");
    msg += file;
    mxERROR(msg.c_str());
  }
  f.close();
}

// The pointers to caffe::Solver and caffe::Net instances
static vector<shared_ptr<Solver<float> > > solvers_;
static vector<shared_ptr<Net<float> > > nets_;
// init_key is generated at the beginning and everytime you call reset
static double init_key = static_cast<double>(caffe_rng_rand());

/** -----------------------------------------------------------------
 ** data conversion functions
 **/
// Enum indicates which blob memory to use
enum WhichMemory { DATA, DIFF };

// Copy matlab array to Blob data or diff
static void mx_mat_to_blob(const mxArray* mx_mat, Blob<float>* blob,
    WhichMemory data_or_diff) {
  mxCHECK(blob->count() == mxGetNumberOfElements(mx_mat),
      "number of elements in target blob doesn't match that in input mxArray");
  const float* mat_mem_ptr = reinterpret_cast<const float*>(mxGetData(mx_mat));
  float* blob_mem_ptr = NULL;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    blob_mem_ptr = (data_or_diff == DATA ?
        blob->mutable_cpu_data() : blob->mutable_cpu_diff());
    break;
  case Caffe::GPU:
    blob_mem_ptr = (data_or_diff == DATA ?
        blob->mutable_gpu_data() : blob->mutable_gpu_diff());
    break;
  default:
    mxERROR("Unknown Caffe mode");
  }
  caffe_copy(blob->count(), mat_mem_ptr, blob_mem_ptr);
}

// Copy Blob data or diff to matlab array
static mxArray* blob_to_mx_mat(const Blob<float>* blob,
    WhichMemory data_or_diff) {
  const int num_axes = blob->num_axes();
  vector<mwSize> dims(num_axes);
  for (int blob_axis = 0, mat_axis = num_axes - 1; blob_axis < num_axes;
       ++blob_axis, --mat_axis) {
    dims[mat_axis] = static_cast<mwSize>(blob->shape(blob_axis));
  }
  // matlab array needs to have at least one dimension, convert scalar to 1-dim
  if (num_axes == 0) {
    dims.push_back(1);
  }
  mxArray* mx_mat =
      mxCreateNumericArray(dims.size(), dims.data(), mxSINGLE_CLASS, mxREAL);
  float* mat_mem_ptr = reinterpret_cast<float*>(mxGetData(mx_mat));
  const float* blob_mem_ptr = NULL;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    blob_mem_ptr = (data_or_diff == DATA ? blob->cpu_data() : blob->cpu_diff());
    break;
  case Caffe::GPU:
    blob_mem_ptr = (data_or_diff == DATA ? blob->gpu_data() : blob->gpu_diff());
    break;
  default:
    mxERROR("Unknown Caffe mode");
  }
  caffe_copy(blob->count(), blob_mem_ptr, mat_mem_ptr);
  return mx_mat;
}

// Convert vector<int> to matlab row vector
static mxArray* int_vec_to_mx_vec(const vector<int>& int_vec) {
  mxArray* mx_vec = mxCreateDoubleMatrix(int_vec.size(), 1, mxREAL);
  double* vec_mem_ptr = mxGetPr(mx_vec);
  for (int i = 0; i < int_vec.size(); i++) {
    vec_mem_ptr[i] = static_cast<double>(int_vec[i]);
  }
  return mx_vec;
}

// Convert vector<string> to matlab cell vector of strings
static mxArray* str_vec_to_mx_strcell(const vector<std::string>& str_vec) {
  mxArray* mx_strcell = mxCreateCellMatrix(str_vec.size(), 1);
  for (int i = 0; i < str_vec.size(); i++) {
    mxSetCell(mx_strcell, i, mxCreateString(str_vec[i].c_str()));
  }
  return mx_strcell;
}

/** -----------------------------------------------------------------
 ** handle and pointer conversion functions
 ** a handle is a struct array with the following fields
 **   (uint64) ptr      : the pointer to the C++ object
 **   (double) init_key : caffe initialization key
 **/
// Convert a handle in matlab to a pointer in C++. Check if init_key matches
template <typename T>
static T* handle_to_ptr(const mxArray* mx_handle) {
  mxArray* mx_ptr = mxGetField(mx_handle, 0, "ptr");
  mxArray* mx_init_key = mxGetField(mx_handle, 0, "init_key");
  mxCHECK(mxIsUint64(mx_ptr), "pointer type must be uint64");
  mxCHECK(mxGetScalar(mx_init_key) == init_key,
      "Could not convert handle to pointer due to invalid init_key. "
      "The object might have been cleared.");
  return reinterpret_cast<T*>(*reinterpret_cast<uint64_t*>(mxGetData(mx_ptr)));
}

// Create a handle struct vector, without setting up each handle in it
template <typename T>
static mxArray* create_handle_vec(int ptr_num) {
  const int handle_field_num = 2;
  const char* handle_fields[handle_field_num] = { "ptr", "init_key" };
  return mxCreateStructMatrix(ptr_num, 1, handle_field_num, handle_fields);
}

// Set up a handle in a handle struct vector by its index
template <typename T>
static void setup_handle(const T* ptr, int index, mxArray* mx_handle_vec) {
  mxArray* mx_ptr = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *reinterpret_cast<uint64_t*>(mxGetData(mx_ptr)) =
      reinterpret_cast<uint64_t>(ptr);
  mxSetField(mx_handle_vec, index, "ptr", mx_ptr);
  mxSetField(mx_handle_vec, index, "init_key", mxCreateDoubleScalar(init_key));
}

// Convert a pointer in C++ to a handle in matlab
template <typename T>
static mxArray* ptr_to_handle(const T* ptr) {
  mxArray* mx_handle = create_handle_vec<T>(1);
  setup_handle(ptr, 0, mx_handle);
  return mx_handle;
}

// Convert a vector of shared_ptr in C++ to handle struct vector
template <typename T>
static mxArray* ptr_vec_to_handle_vec(const vector<shared_ptr<T> >& ptr_vec) {
  mxArray* mx_handle_vec = create_handle_vec<T>(ptr_vec.size());
  for (int i = 0; i < ptr_vec.size(); i++) {
    setup_handle(ptr_vec[i].get(), i, mx_handle_vec);
  }
  return mx_handle_vec;
}

/** -----------------------------------------------------------------
 ** matlab command functions: caffe_(api_command, arg1, arg2, ...)
 **/
// Usage: caffe_('get_solver', solver_file);
static void get_solver(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsChar(prhs[0]),
      "Usage: caffe_('get_solver', solver_file)");
  char* solver_file = mxArrayToString(prhs[0]);
  mxCHECK_FILE_EXIST(solver_file);
  shared_ptr<Solver<float> > solver(new caffe::SGDSolver<float>(solver_file));
  solvers_.push_back(solver);
  plhs[0] = ptr_to_handle<Solver<float> >(solver.get());
  mxFree(solver_file);
}

// Usage: caffe_('solver_get_attr', hSolver)
static void solver_get_attr(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('solver_get_attr', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  const int solver_attr_num = 2;
  const char* solver_attrs[solver_attr_num] = { "hNet_net", "hNet_test_nets" };
  mxArray* mx_solver_attr = mxCreateStructMatrix(1, 1, solver_attr_num,
      solver_attrs);
  mxSetField(mx_solver_attr, 0, "hNet_net",
      ptr_to_handle<Net<float> >(solver->net().get()));
  mxSetField(mx_solver_attr, 0, "hNet_test_nets",
      ptr_vec_to_handle_vec<Net<float> >(solver->test_nets()));
  plhs[0] = mx_solver_attr;
}

// Usage: caffe_('solver_get_iter', hSolver)
static void solver_get_iter(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('solver_get_iter', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  plhs[0] = mxCreateDoubleScalar(solver->iter());
}

// Usage: caffe_('solver_restore', hSolver, snapshot_file)
static void solver_restore(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('solver_restore', hSolver, snapshot_file)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  char* snapshot_file = mxArrayToString(prhs[1]);
  mxCHECK_FILE_EXIST(snapshot_file);
  solver->Restore(snapshot_file);
  mxFree(snapshot_file);
}

// Usage: caffe_('solver_solve', hSolver)
static void solver_solve(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('solver_solve', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  solver->Solve();
}

// Usage: caffe_('solver_step', hSolver, iters)
static void solver_step(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsDouble(prhs[1]),
      "Usage: caffe_('solver_step', hSolver, iters)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  int iters = mxGetScalar(prhs[1]);
  solver->Step(iters);
}

// Usage: caffe_('get_net', model_file, phase_name)
static void get_net(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsChar(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('get_net', model_file, phase_name)");
  char* model_file = mxArrayToString(prhs[0]);
  char* phase_name = mxArrayToString(prhs[1]);
  mxCHECK_FILE_EXIST(model_file);
  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
      phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
      phase = TEST;
  } else {
    mxERROR("Unknown phase");
  }
  shared_ptr<Net<float> > net(new caffe::Net<float>(model_file, phase));
  nets_.push_back(net);
  plhs[0] = ptr_to_handle<Net<float> >(net.get());
  mxFree(model_file);
  mxFree(phase_name);
}

// Usage: caffe_('net_get_attr', hNet)
static void net_get_attr(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_get_attr', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  const int net_attr_num = 6;
  const char* net_attrs[net_attr_num] = { "hLayer_layers", "hBlob_blobs",
      "input_blob_indices", "output_blob_indices", "layer_names", "blob_names"};
  mxArray* mx_net_attr = mxCreateStructMatrix(1, 1, net_attr_num,
      net_attrs);
  mxSetField(mx_net_attr, 0, "hLayer_layers",
      ptr_vec_to_handle_vec<Layer<float> >(net->layers()));
  mxSetField(mx_net_attr, 0, "hBlob_blobs",
      ptr_vec_to_handle_vec<Blob<float> >(net->blobs()));
  mxSetField(mx_net_attr, 0, "input_blob_indices",
      int_vec_to_mx_vec(net->input_blob_indices()));
  mxSetField(mx_net_attr, 0, "output_blob_indices",
      int_vec_to_mx_vec(net->output_blob_indices()));
  mxSetField(mx_net_attr, 0, "layer_names",
      str_vec_to_mx_strcell(net->layer_names()));
  mxSetField(mx_net_attr, 0, "blob_names",
      str_vec_to_mx_strcell(net->blob_names()));
  plhs[0] = mx_net_attr;
}

// Usage: caffe_('net_forward', hNet)
static void net_forward(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_forward', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->ForwardPrefilled();
}

// Usage: caffe_('net_backward', hNet)
static void net_backward(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_backward', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->Backward();
  //LOG(INFO) << "end of matcaffe net_backward";
}

// Usage: caffe_('net_copy_from', hNet, weights_file)
static void net_copy_from(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('net_copy_from', hNet, weights_file)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  char* weights_file = mxArrayToString(prhs[1]);
  mxCHECK_FILE_EXIST(weights_file);
  net->CopyTrainedLayersFrom(weights_file);
  mxFree(weights_file);
}

// Usage: caffe_('net_reshape', hNet)
static void net_reshape(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_reshape', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->Reshape();
}

// Usage: caffe_('net_save', hNet, save_file)
static void net_save(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('net_save', hNet, save_file)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  char* weights_file = mxArrayToString(prhs[1]);
  NetParameter net_param;
  net->ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, weights_file);
  mxFree(weights_file);
}

// Usage: caffe_('layer_get_attr', hLayer)
static void layer_get_attr(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('layer_get_attr', hLayer)");
  Layer<float>* layer = handle_to_ptr<Layer<float> >(prhs[0]);
  const int layer_attr_num = 1;
  const char* layer_attrs[layer_attr_num] = { "hBlob_blobs" };
  mxArray* mx_layer_attr = mxCreateStructMatrix(1, 1, layer_attr_num,
      layer_attrs);
  mxSetField(mx_layer_attr, 0, "hBlob_blobs",
      ptr_vec_to_handle_vec<Blob<float> >(layer->blobs()));
  plhs[0] = mx_layer_attr;
}

// Usage: caffe_('layer_get_type', hLayer)
static void layer_get_type(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('layer_get_type', hLayer)");
  Layer<float>* layer = handle_to_ptr<Layer<float> >(prhs[0]);
  plhs[0] = mxCreateString(layer->type());
}

// Usage: caffe_('blob_get_shape', hBlob)
static void blob_get_shape(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('blob_get_shape', hBlob)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  const int num_axes = blob->num_axes();
  mxArray* mx_shape = mxCreateDoubleMatrix(1, num_axes, mxREAL);
  double* shape_mem_mtr = mxGetPr(mx_shape);
  for (int blob_axis = 0, mat_axis = num_axes - 1; blob_axis < num_axes;
       ++blob_axis, --mat_axis) {
    shape_mem_mtr[mat_axis] = static_cast<double>(blob->shape(blob_axis));
  }
  plhs[0] = mx_shape;
}

// Usage: caffe_('blob_reshape', hBlob, new_shape)
static void blob_reshape(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsDouble(prhs[1]),
      "Usage: caffe_('blob_reshape', hBlob, new_shape)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  const mxArray* mx_shape = prhs[1];
  double* shape_mem_mtr = mxGetPr(mx_shape);
  const int num_axes = mxGetNumberOfElements(mx_shape);
  vector<int> blob_shape(num_axes);
  for (int blob_axis = 0, mat_axis = num_axes - 1; blob_axis < num_axes;
       ++blob_axis, --mat_axis) {
    blob_shape[blob_axis] = static_cast<int>(shape_mem_mtr[mat_axis]);
  }
  blob->Reshape(blob_shape);
}

// Usage: caffe_('blob_get_data', hBlob)
static void blob_get_data(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('blob_get_data', hBlob)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  plhs[0] = blob_to_mx_mat(blob, DATA);
}

static void my_get_data(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('blob_get_data', hBlob)");
 // Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
    plhs[0] = blob_to_mx_mat(Layer<float>::input1[1], DATA);
    plhs[1] = blob_to_mx_mat(Layer<float>::input1[2], DATA);
    plhs[2] = blob_to_mx_mat(Layer<float>::input1[3], DATA);
    plhs[3] = blob_to_mx_mat(Layer<float>::input1[4], DATA);
    plhs[4] = blob_to_mx_mat(Layer<float>::input1[9], DATA);
    plhs[5] = blob_to_mx_mat(Layer<float>::input1[10], DATA);

   Blob<float>* tmp=new Blob<float>();
   tmp->Reshape(1,1,11*11,46*46);;
  caffe_copy(tmp->count(),Layer<float>::seventhlayer_tmp[0]->mutable_gpu_data(),tmp->mutable_gpu_data());
  plhs[6] = blob_to_mx_mat(tmp, DATA);
}

// Usage: caffe_('blob_set_data', hBlob, new_data)
static void blob_set_data(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
      "Usage: caffe_('blob_set_data', hBlob, new_data)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  mx_mat_to_blob(prhs[1], blob, DATA);
}
static void my_set_data2(MEX_ARGS){
    mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
    const int * dims = static_cast<const int*>(mxGetDimensions(prhs[1]));
    Layer<float>::input1[7]->Reshape(1,dims[2],dims[1],dims[0]);
    mx_mat_to_blob(prhs[1],Layer<float>::input1[7], DATA);
    const  mxArray* mx_mat1=prhs[2];
    const float* lambda1 = reinterpret_cast<const float*>(mxGetData(mx_mat1));
    const  mxArray* mx_mat2=prhs[3];
    const float* lambda2 = reinterpret_cast<const float*>(mxGetData(mx_mat2));
    Layer<float>::input1[8]->Reshape(1,1,1,2);
    float * lambda=Layer<float>::input1[8]->mutable_cpu_data();
    lambda[0]=lambda1[0]; lambda[1]=lambda2[0];

    mx_mat_to_blob(prhs[4],Layer<float>::input1[9],DATA);//用于输出第一层权重

    mx_mat_to_blob(prhs[5],Layer<float>::input1[10],DATA);

}


static void my_set_data3(MEX_ARGS){
    mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
    const int * dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
    const int * dims2 = static_cast<const int*>(mxGetDimensions(prhs[2]));    
    const int * dims3 = static_cast<const int*>(mxGetDimensions(prhs[3]));
    const int * dims4 = static_cast<const int*>(mxGetDimensions(prhs[4]));
    const int * dims5 = static_cast<const int*>(mxGetDimensions(prhs[5]));
    const int * dims6 = static_cast<const int*>(mxGetDimensions(prhs[6]));
    const int * dims7 = static_cast<const int*>(mxGetDimensions(prhs[7]));
}


static void my_set_data7(MEX_ARGS) {
mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
Blob<float>* template_x=new Blob<float>();
template_x->Reshape(1,1,11,11);
Blob<float>* template_x1=new Blob<float>();
template_x1->Reshape(1,1,11,11);
Blob<float>* template_y=new Blob<float>();
template_y->Reshape(1,1,11,11);
Blob<float>* template_y1=new Blob<float>();
template_y1->Reshape(1,1,11,11);
mx_mat_to_blob(prhs[1],template_x, DATA);
mx_mat_to_blob(prhs[2],template_x1, DATA);
mx_mat_to_blob(prhs[3],template_y, DATA);
mx_mat_to_blob(prhs[4],template_y1, DATA);


Layer<float>::seventhlayer_tmp.resize(1);
Layer<float>::seventhlayer_tmp[0].reset(new Blob<float>());
Layer<float>::seventhlayer_tmp[0]->Reshape(1,1,11*11,46*46);

Layer<float>::seventhlayer_tmp1.resize(1);
Layer<float>::seventhlayer_tmp1[0].reset(new Blob<float>());
Layer<float>::seventhlayer_tmp1[0]->Reshape(1,1,4,46*46); //按列存储x x^2 y y^2

Layer<float>::seventhlayer_template_x.resize(1);
Layer<float>::seventhlayer_template_x[0].reset(new Blob<float>());
Layer<float>::seventhlayer_template_x[0]->Reshape(1,1,11,11);
Layer<float>::seventhlayer_template_x1.resize(1);
Layer<float>::seventhlayer_template_x1[0].reset(new Blob<float>());
Layer<float>::seventhlayer_template_x1[0]->Reshape(1,1,11,11);
Layer<float>::seventhlayer_template_y.resize(1);
Layer<float>::seventhlayer_template_y[0].reset(new Blob<float>());
Layer<float>::seventhlayer_template_y[0]->Reshape(1,1,11,11);
Layer<float>::seventhlayer_template_y1.resize(1);
Layer<float>::seventhlayer_template_y1[0].reset(new Blob<float>());
Layer<float>::seventhlayer_template_y1[0]->Reshape(1,1,11,11);

Layer<float>::seventhlayer_col_buff.resize(1);
Layer<float>::seventhlayer_col_buff[0].reset(new Blob<float>());
Layer<float>::seventhlayer_col_buff[0]->Reshape(1,1,11*11,46*46);

switch (Caffe::mode()) {
            case Caffe::CPU: {
            caffe_copy(template_x->count(),template_x->mutable_cpu_data(),Layer<float>::seventhlayer_template_x[0]->mutable_cpu_data());
            caffe_copy(template_x1->count(),template_x1->mutable_cpu_data(),Layer<float>::seventhlayer_template_x1[0]->mutable_cpu_data());
            caffe_copy(template_y->count(),template_y->mutable_cpu_data(),Layer<float>::seventhlayer_template_y[0]->mutable_cpu_data());
            caffe_copy(template_y1->count(),template_y1->mutable_cpu_data(),Layer<float>::seventhlayer_template_y1[0]->mutable_cpu_data()); 
            }
            break;
            case Caffe::GPU: {
            caffe_copy(template_x->count(),template_x->mutable_gpu_data(),Layer<float>::seventhlayer_template_x[0]->mutable_gpu_data());
            caffe_copy(template_x1->count(),template_x1->mutable_gpu_data(),Layer<float>::seventhlayer_template_x1[0]->mutable_gpu_data());
            caffe_copy(template_y->count(),template_y->mutable_gpu_data(),Layer<float>::seventhlayer_template_y[0]->mutable_gpu_data());
            caffe_copy(template_y1->count(),template_y1->mutable_gpu_data(),Layer<float>::seventhlayer_template_y1[0]->mutable_gpu_data());
            }
            break;
            default:
            mxERROR("Unknown Caffe mode");
        }
delete template_x; delete template_x1; delete template_y; delete template_y1;
}

static void my_set_data5(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
   const  mxArray* mx_mat1=prhs[1];
   const float* channel = reinterpret_cast<const float*>(mxGetData(mx_mat1));
   const  mxArray* mx_mat2=prhs[2];
   const float* kernel_h = reinterpret_cast<const float*>(mxGetData(mx_mat2));
   const  mxArray* mx_mat3=prhs[3];
   const float* kernel_w = reinterpret_cast<const float*>(mxGetData(mx_mat3));
   const  mxArray* mx_mat4=prhs[4];
   const float* height = reinterpret_cast<const float*>(mxGetData(mx_mat4));
   const  mxArray* mx_mat5=prhs[5];
   const float* width = reinterpret_cast<const float*>(mxGetData(mx_mat5));
   const  mxArray* mx_mat6=prhs[6];
   const float* pad_h = reinterpret_cast<const float*>(mxGetData(mx_mat6));
   const  mxArray* mx_mat7=prhs[7];
   const float* pad_w = reinterpret_cast<const float*>(mxGetData(mx_mat7)); 
   const  mxArray* mx_mat8=prhs[8];
   const float* stride_h = reinterpret_cast<const float*>(mxGetData(mx_mat8));
   const  mxArray* mx_mat9=prhs[9];
   const float* stride_w = reinterpret_cast<const float*>(mxGetData(mx_mat9));  


   int out_h = (height[0] - kernel_h[0]+2*pad_h[0])/stride_h[0] +1;
   int out_w = (width[0] - kernel_w[0]+2*pad_w[0])/stride_w[0] +1;
  float aaa=(height[0] - kernel_h[0]+2*pad_h[0])/stride_h[0] +1;
    printf("aaa is %f %d\n\n\n",aaa,out_h);
    printf("in caffe.cpp out_h out_w %f %f %f %f %f %f\n\n\n",height[0],width[0],kernel_h[0],kernel_w[0],pad_h[0],pad_w[0]);

    Layer<float>::fifthlayer_col_buff.resize(1);
   Layer<float>::fifthlayer_col_buff[0].reset(new Blob<float>());
   Layer<float>::fifthlayer_col_buff[0]->Reshape(1,1,channel[0]*kernel_h[0]/3*kernel_w[0]/3,out_h*out_w);
   
   const int* dims = static_cast<const int*>(mxGetDimensions(prhs[10]));

   Layer<float>::fifthlayer_tmp.resize(1);
   Layer<float>::fifthlayer_tmp[0].reset(new Blob<float>());
   Layer<float>::fifthlayer_tmp[0]->Reshape(9,dims[2],dims[1],dims[0]); 
   Blob<float>* tmp=new Blob<float>();
   tmp->Reshape(9,dims[2],dims[1],dims[0]);
   mx_mat_to_blob(prhs[10],tmp, DATA);
   caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::fifthlayer_tmp[0]->mutable_gpu_data());

    Layer<float>::fifthlayer_template1.resize(1);
    Layer<float>::fifthlayer_template1[0].reset(new Blob<float>());
    Layer<float>::fifthlayer_template1[0]->Reshape(1,1,kernel_h[0]/3*kernel_w[0]/3*channel[0],9);
    Layer<float>::fifthlayer_data_im.resize(1);
    Layer<float>::fifthlayer_data_im[0].reset(new Blob<float>());
    Layer<float>::fifthlayer_data_im[0]->Reshape(1,channel[0],kernel_h[0],kernel_w[0]);
    Layer<float>::fifthlayer_data.resize(1);
    Layer<float>::fifthlayer_data[0].reset(new Blob<float>());
    Layer<float>::fifthlayer_data[0]->Reshape(1,1,1,11);
    float *data=Layer<float>::fifthlayer_data[0]->mutable_cpu_data();
     data[0]=channel[0]; data[1]=height[0]; data[2]=width[0]; data[5]=kernel_h[0]; data[6]=kernel_w[0];
     data[7]=pad_h[0]; data[8]=pad_w[0]; data[9]=stride_h[0]; data[10]=stride_w[0];

    Layer<float>::sixthlayer_col_buff.resize(1);
    Layer<float>::sixthlayer_col_buff[0].reset(new Blob<float>());
    Layer<float>::sixthlayer_col_buff[0]->Reshape(1,1,channel[0]*kernel_h[0]*kernel_w[0],out_h*out_w);

    Layer<float>::sixthlayer_col_buff1.resize(1);
    Layer<float>::sixthlayer_col_buff1[0].reset(new Blob<float>());
    Layer<float>::sixthlayer_col_buff1[0]->Reshape(1,1,channel[0]*kernel_h[0]*kernel_w[0],1);

    Layer<float>::fifthlayer_tmp4.resize(1);
    Layer<float>::fifthlayer_tmp4[0].reset(new Blob<float>());
    Layer<float>::fifthlayer_tmp4[0]->Reshape(1,1,channel[0]*kernel_h[0]/3*kernel_w[0]/3,9);

    Layer<float>::fifthlayer_tmp5.resize(1);
    Layer<float>::fifthlayer_tmp5[0].reset(new Blob<float>());
    Layer<float>::fifthlayer_tmp5[0]->Reshape(1,channel[0],kernel_h[0],kernel_w[0]);

delete tmp; 



}

static void my_set_data8(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");

    const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1]));
 
  float *  data= Layer<float>::input[0]->mutable_cpu_data();
   //  data[0]=channel[0]; data[1]=height[0]; data[2]=width[0]; data[3]=height_in[0];
  //   data[4]=width_in[0]; data[5]=kernel_h[0]; data[6]=kernel_w[0];
  //   data[7]=pad_h[0]; data[8]=pad_w[0]; data[9]=stride_h[0]; data[10]=stride_w[0];
  int kernel_h=data[5]; int kernel_w=data[6]; int height=dims[1]; int width=dims[0];
  Blob<float>* tmp=new Blob<float>();
  tmp->Reshape(9,dims[2],dims[1],dims[0]);
  mx_mat_to_blob(prhs[1],tmp, DATA);
  
   Layer<float>::eighthlayer_tmp.resize(1);
   Layer<float>::eighthlayer_tmp[0].reset(new Blob<float>());
   Layer<float>::eighthlayer_tmp[0]->Reshape(9,dims[2],dims[1],dims[0]);
  //求取out_h及out_w
    int out_h_second = (tmp->height() - kernel_h/3)/1 +1;
    int out_w_second = (tmp->width() - kernel_w/3)/1 +1; 
    printf("out_h and out_w are %d %d\n\n",out_h_second,out_w_second);

    Layer<float>::eighthlayer_col_buff.resize(1);
    Layer<float>::eighthlayer_col_buff[0].reset(new Blob<float>());
    Layer<float>::eighthlayer_col_buff[0]->Reshape(1,1,kernel_h*kernel_w/9*512,out_h_second*out_w_second);

    Layer<float>::eighthlayer_tmp1.resize(1);
    Layer<float>::eighthlayer_tmp1[0].reset(new Blob<float>());
    Layer<float>::eighthlayer_tmp1[0]->Reshape(1,1,9,out_h_second*out_w_second);


    delete tmp;
}


static void my_set_data9(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");

    const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1]));
 
  float *  data= Layer<float>::input[0]->mutable_cpu_data();
  int kernel_h=data[5]; int kernel_w=data[6]; int height=dims[1]; int width=dims[0];
  Blob<float>* tmp=new Blob<float>();
  tmp->Reshape(9,dims[2],dims[1],dims[0]);
  mx_mat_to_blob(prhs[1],tmp, DATA);
  
   Layer<float>::ninthlayer_tmp.resize(1);
   Layer<float>::ninthlayer_tmp[0].reset(new Blob<float>());
   Layer<float>::ninthlayer_tmp[0]->Reshape(9,dims[2],dims[1],dims[0]);
  
   caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::ninthlayer_tmp[0]->mutable_gpu_data());

  //求取out_h及out_w
    int out_h_second = (tmp->height() - kernel_h/3)/1 +1;
    int out_w_second = (tmp->width() - kernel_w/3)/1 +1; 
    printf("out_h and out_w are %d %d\n\n",out_h_second,out_w_second);

    Layer<float>::ninthlayer_col_buff.resize(1);
    Layer<float>::ninthlayer_col_buff[0].reset(new Blob<float>());
    Layer<float>::ninthlayer_col_buff[0]->Reshape(1,1,kernel_h*kernel_w/9*512,out_h_second*out_w_second);

    Layer<float>::ninthlayer_tmp1.resize(1);
    Layer<float>::ninthlayer_tmp1[0].reset(new Blob<float>());
    Layer<float>::ninthlayer_tmp1[0]->Reshape(9,1,1,out_h_second*out_w_second);

    Layer<float>::ninthlayer_template_tmp.resize(1);
    Layer<float>::ninthlayer_template_tmp[0].reset(new Blob<float>());
    Layer<float>::ninthlayer_template_tmp[0]->Reshape(1,1,kernel_h*kernel_w/9*512,1);
 
    Layer<float>::ninthlayer_col_buff1.resize(1);
    Layer<float>::ninthlayer_col_buff1[0].reset(new Blob<float>());
    Layer<float>::ninthlayer_col_buff1[0]->Reshape(1,1,11*11,out_h_second*out_w_second);

    Layer<float>::ninthlayer_tmp2.resize(1);
    Layer<float>::ninthlayer_tmp2[0].reset(new Blob<float>());
    Layer<float>::ninthlayer_tmp2[0]->Reshape(1,1,11*11,out_h_second*out_w_second);
 
    Layer<float>::ninthlayer_tmp3.resize(1);
    Layer<float>::ninthlayer_tmp3[0].reset(new Blob<float>());
    Layer<float>::ninthlayer_tmp3[0]->Reshape(1,1,4,out_h_second*out_w_second);

    Layer<float>::ninthlayer_tmp4.resize(1);
    Layer<float>::ninthlayer_tmp4[0].reset(new Blob<float>());
    Layer<float>::ninthlayer_tmp4[0]->Reshape(1,1,kernel_h*kernel_w/9,9);

    delete tmp;
}


static void set_momentum(MEX_ARGS) {
mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
    const  mxArray* mx_mat1=prhs[1];
    const float*  momentum= reinterpret_cast<const float*>(mxGetData(mx_mat1));
    Layer<float>::momentum.resize(1);
    Layer<float>::momentum[0].reset(new Blob<float>());
    Layer<float>::momentum[0]->Reshape(1,1,1,1);
    float* data=Layer<float>::momentum[0]->mutable_cpu_data();
    data[0]=momentum[0];
}

 static void my_set_data(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
      // vector<shared_ptr<Blob<float> > > tmp=Layer<float>::get_input();
//printf("test11\n\n\n\n\n\n");
   //  Layer<float>::get_input();
     //
     //
     const int* dims = static_cast<const int*>(mxGetDimensions(prhs[12]));
     //
    const  mxArray* mx_mat1=prhs[1];
    const float* channel = reinterpret_cast<const float*>(mxGetData(mx_mat1));
 const mxArray *   mx_mat2=prhs[2];
    const float* height = reinterpret_cast<const float*>(mxGetData(mx_mat2));
 const mxArray *  mx_mat3=prhs[3];
    const float* width = reinterpret_cast<const float*>(mxGetData(mx_mat3));
  const mxArray *   mx_mat4=prhs[4];
    const float* height_in = reinterpret_cast<const float*>(mxGetData(mx_mat4));
 const mxArray * mx_mat5= prhs[5];
const float* width_in = reinterpret_cast<const float*>(mxGetData(mx_mat5));
const mxArray * mx_mat6= prhs[6];
const float* kernel_h = reinterpret_cast<const float*>(mxGetData(mx_mat6));
const mxArray * mx_mat7= prhs[7];
const float* kernel_w = reinterpret_cast<const float*>(mxGetData(mx_mat7));
const mxArray * mx_mat8= prhs[8];
const float* pad_h = reinterpret_cast<const float*>(mxGetData(mx_mat8));
const mxArray * mx_mat9= prhs[9];
const float* pad_w = reinterpret_cast<const float*>(mxGetData(mx_mat9));
const mxArray * mx_mat10= prhs[10];
const float* stride_h = reinterpret_cast<const float*>(mxGetData(mx_mat10));
const mxArray * mx_mat11= prhs[11];
const float* stride_w = reinterpret_cast<const float*>(mxGetData(mx_mat11));


     Layer<float>::input.resize(20);
     Layer<float>::input[0].reset(new Blob<float>());
     Layer<float>::input[0]->Reshape(1,1,1,11);
     float *  data= Layer<float>::input[0]->mutable_cpu_data();
     data[0]=channel[0]; data[1]=height[0]; data[2]=width[0]; data[3]=height_in[0];
     data[4]=width_in[0]; data[5]=kernel_h[0]; data[6]=kernel_w[0];
     data[7]=pad_h[0]; data[8]=pad_w[0]; data[9]=stride_h[0]; data[10]=stride_w[0];
    int out_h = (height[0] - kernel_h[0]+2*pad_h[0])/stride_h[0] +1;
    int out_w = (width[0] - kernel_w[0]+2*pad_w[0])/stride_w[0] +1;
    Layer<float>::input[1].reset(new Blob<float>());
    Layer<float>::input[1]->Reshape(1,1,(int) kernel_h[0]*(int) kernel_w[0]*(int)channel[0],out_h*out_w);
    // printf("this size is%d  %d\n\n\n\n",Layer<float>::input[1]->height(),Layer<float>::input[1]->width());
    Layer<float>::input1[0]->Reshape(9, dims[2], dims[1], dims[0]);

    Layer<float>::input[2].reset(new Blob<float>());
    Layer<float>::input[2]->Reshape(1,1,kernel_h[0]*kernel_w[0]*data[0],1); // 用于存储第一层中间变量 for test
    Layer<float>::input[3].reset(new Blob<float>());
    Layer<float>::input[3]->Reshape(1,data[0],data[5],data[6]); // 用于存储第一层中间变量 for test
    Layer<float>::input[4].reset(new Blob<float>());
    Layer<float>::input[4]->Reshape(1,1,kernel_h[0]/3*kernel_w[0]/3*channel[0],9);
    Layer<float>::input[5].reset(new Blob<float>());
    Layer<float>::input[5]->Reshape(1,channel[0],kernel_h[0],kernel_w[0]);
    Layer<float>::input[6].reset(new Blob<float>());
    Layer<float>::input[6]->Reshape(1,channel[0],kernel_h[0],kernel_w[0]);
    mx_mat_to_blob(prhs[12],Layer<float>::input1[0], DATA); 
 


    int out_h_second = (Layer<float>::input1[0]->height() - kernel_h[0]/3)/stride_h[0] +1;
    int out_w_second = (Layer<float>::input1[0]->width() - kernel_w[0]/3)/stride_w[0] +1;
    Layer<float>::input[7].reset(new Blob<float>());
    Layer<float>::input[7]->Reshape(1,1,9,out_h_second*out_w_second);

    Layer<float>::input[8].reset(new Blob<float>());
    Layer<float>::input[8]->Reshape(1,1,channel[0]*kernel_h[0]/3*kernel_w[0]/3,out_h_second*out_w_second);

    Layer<float>::input[9].reset(new Blob<float>());
    Layer<float>::input[9]->Reshape(1,1,channel[0]*kernel_h[0]/3*kernel_w[0]/3,9);

    Layer<float>::input[10].reset(new Blob<float>());
    Layer<float>::input[10]->Reshape(1,1,9,out_h_second*out_w_second);

    Layer<float>::input[11].reset(new Blob<float>());
    Layer<float>::input[11]->Reshape(1,1,channel[0]*kernel_h[0]/3*kernel_w[0]/3,out_h_second*out_w_second);

    Layer<float>::input[12].reset(new Blob<float>());
    Layer<float>::input[12]->Reshape(1,1,out_h_second*out_w_second,9);

    Layer<float>::input[13].reset(new Blob<float>());
    Layer<float>::input[13]->Reshape(1,1,channel[0]*kernel_h[0]/3*kernel_w[0]/3,9);

    Layer<float>::input[14].reset(new Blob<float>());//临时存储第一层权重
    Layer<float>::input[14]->Reshape(1,1,out_h*out_w,1);

    Layer<float>::input[15].reset(new Blob<float>());//临时存储第二层权重
    Layer<float>::input[15]->Reshape(1,1,1,81);

    Layer<float>::input[16].reset(new Blob<float>());//临时存储模板
    Layer<float>::input[16]->Reshape(1,channel[0],height[0],width[0]);

    Layer<float>::input1[9]->Reshape(1,1,out_h*out_w,1); //用于输出第一层权重

    Layer<float>::input1[10]->Reshape(1,1,1,81); //用于输出第二层权重
  
    
    Layer<float>::input1[1]->Reshape(1,1,1,out_h*out_w);

    
    Layer<float>::input1[2]->Reshape(1,1,1,81);

    Layer<float>::input1[3]->Reshape(1,1,9,out_h_second*out_w_second);

    Layer<float>::input1[4]->Reshape(1,channel[0],kernel_h[0],kernel_w[0]);

    Layer<float>::input1[5]->Reshape(1,1,channel[0]*kernel_h[0]*kernel_w[0],1);

    Layer<float>::input1[6]->Reshape(1,channel[0],kernel_h[0],kernel_w[0]);
 
    Layer<float>::input1[11]->Reshape(1,1,channel[0]*kernel_h[0]*kernel_w[0],out_h_second*out_w_second);

    Layer<float>::padded_feature.resize(1);
    Layer<float>::padded_feature[0].reset(new Blob<float>());
    Layer<float>::padded_feature[0]->Reshape(1,512,height[0]+2*pad_h[0],width[0]+2*pad_w[0]);

    Layer<float>::cos_window.resize(1);
    Layer<float>::cos_window[0].reset(new Blob<float>());
    Layer<float>::cos_window[0]->Reshape(1,1,46,46);
 
    Blob<float>* tmp=new Blob<float>();
    tmp->Reshape(1,1,46,46);   
    mx_mat_to_blob(prhs[13],tmp, DATA);

     switch (Caffe::mode()) {
            case Caffe::CPU: {
            caffe_copy(tmp->count(),tmp->mutable_cpu_data(),Layer<float>::cos_window[0]->mutable_cpu_data()); 
            }
            break;
            case Caffe::GPU: {
            caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::cos_window[0]->mutable_gpu_data());
            }
            break;
            default:
            mxERROR("Unknown Caffe mode");
        }
    delete tmp;

    Layer<float>::masked_feature.resize(1);
    Layer<float>::masked_feature[0].reset(new Blob<float>());
    Layer<float>::masked_feature[0]->Reshape(1,512,46,46);


    Layer<float>::first_layer_col.resize(1);
    Layer<float>::first_layer_col[0].reset(new Blob<float>());
    Layer<float>::first_layer_col[0]->Reshape(1,1,kernel_h[0]*kernel_w[0]*41,out_h*out_w);


   Layer<float>::first_layer_top_data.resize(1);
   Layer<float>::first_layer_top_data[0].reset(new Blob<float>());
   Layer<float>::first_layer_top_data[0]->Reshape(1,1,kernel_h[0]*kernel_w[0]*41,1);

   Layer<float>::second_layer_template1.resize(1);
   Layer<float>::second_layer_template1[0].reset(new Blob<float>());
   Layer<float>::second_layer_template1[0]->Reshape(1,1,kernel_h[0]/3*kernel_w[0]/3*41,9); 
  
   Layer<float>::second_layer_col_buff.resize(1);
   Layer<float>::second_layer_col_buff[0].reset(new Blob<float>());
   Layer<float>::second_layer_col_buff[0]->Reshape(1,1,kernel_h[0]/3*kernel_w[0]/3*41,out_h_second*out_w_second); 

   Layer<float>::second_layer_tmp1.resize(1);
   Layer<float>::second_layer_tmp1[0].reset(new Blob<float>());
   Layer<float>::second_layer_tmp1[0]->Reshape(1,1,9,out_h_second*out_w_second);


   Layer<float>::second_layer_top_data.resize(1);
   Layer<float>::second_layer_top_data[0].reset(new Blob<float>());
   Layer<float>::second_layer_top_data[0]->Reshape(1,1,1,out_h_second*out_w_second); 

   Layer<float>::second_layer_weight_diff.resize(1);
   Layer<float>::second_layer_weight_diff[0].reset(new Blob<float>());
   Layer<float>::second_layer_weight_diff[0]->Reshape(1,1,1,81); 

   Layer<float>::second_layer_tmp3.resize(1);
   Layer<float>::second_layer_tmp3[0].reset(new Blob<float>());
   Layer<float>::second_layer_tmp3[0]->Reshape(1,1,out_h_second*out_w_second,9);  

   Layer<float>::second_layer_tmp4.resize(1);
   Layer<float>::second_layer_tmp4[0].reset(new Blob<float>());
   Layer<float>::second_layer_tmp4[0]->Reshape(1,1,kernel_h[0]/3*kernel_w[0]/3*41,9);  
 
   Layer<float>::second_layer_bottom_diff.resize(1);
   Layer<float>::second_layer_bottom_diff[0].reset(new Blob<float>());
   Layer<float>::second_layer_bottom_diff[0]->Reshape(1,1,kernel_h[0]*kernel_w[0]*41,1);  

   Layer<float>::first_layer_weight_diff.resize(1);
   Layer<float>::first_layer_weight_diff[0].reset(new Blob<float>());
   Layer<float>::first_layer_weight_diff[0]->Reshape(1,1,1,out_h*out_w);  

//  int out_h_tmp = (48 - 5+2*2)/1 +1;
//  int out_w_tmp = (48 - 5+2*2)/1 +1;
//Layer<float>::third_layer_col.resize(1);
//Layer<float>::third_layer_col[0].reset(new Blob<float>());
//   Layer<float>::third_layer_col[0]->Reshape(1,1,512*5*5,46*46);
// Layer<float>::third_layer_template_col.resize(1);
// Layer<float>::third_layer_template_col[0].reset(new Blob<float>());

//   Layer<float>::third_layer_template_col[0]->Reshape(1,1,512*5*5,out_h_tmp*out_w_tmp); //使用20个随机遮罩
//    Layer<float>::third_layer_masked_data.resize(1);
// Layer<float>::third_layer_masked_data[0].reset(new Blob<float>());
//   Layer<float>::third_layer_masked_data[0]->Reshape(1,512,48,48); //考虑卷积核为5的情况
//Layer<float>::third_layer_tmp_weight.resize(1);
//Layer<float>::third_layer_tmp_weight[0].reset(new Blob<float>());
//Layer<float>::third_layer_tmp_weight[0]->Reshape(100,512,6,6);
//Layer<float>::third_layer_input_feature.resize(1);
//Layer<float>::third_layer_input_feature[0].reset(new Blob<float>());
//Layer<float>::third_layer_input_feature[0]->Reshape(1,channel[0],46,46);
//Layer<float>::third_layer_tmp_diff.resize(1);
//Layer<float>::third_layer_tmp_diff[0].reset(new Blob<float>());
//Layer<float>::third_layer_tmp_diff[0]->Reshape(1,1,20,channel[0]*5*5);
//Layer<float>::third_layer_first_frame_feature.resize(1);
//Layer<float>::third_layer_first_frame_feature[0].reset(new Blob<float>());
//Layer<float>::third_layer_first_frame_feature[0]->Reshape(1,512,48,48);
//Blob<float>* tmp=new Blob<float>();
//tmp->Reshape(1,channel[0],48,48);
//mx_mat_to_blob(prhs[14],tmp, DATA);
//caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::third_layer_first_frame_feature[0]->mutable_gpu_data());
//delete tmp; 
/*
Blob<float>* rotation_tmp=new Blob<float>();
rotation_tmp->Reshape(9,channel[0],kernel_h[0]/3,kernel_w[0]/3);
mx_mat_to_blob(prhs[15],rotation_tmp, DATA);
Layer<float>::rotation_tmp.resize(1);
Layer<float>::rotation_tmp[0].reset(new Blob<float>());
Layer<float>::rotation_tmp[0]->Reshape(9,channel[0],kernel_h[0]/3,kernel_w[0]/3);

caffe_copy(rotation_tmp->count(),rotation_tmp->mutable_gpu_data(),Layer<float>::rotation_tmp[0]->mutable_gpu_data());
delete rotation_tmp;
*/
 }

 static void my_set_data1(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
      // vector<shared_ptr<Blob<float> > > tmp=Layer<float>::get_input();
//printf("test7\n\n\n\n\n\n");
   //  Layer<float>::get_input();
     //
     //
     //
  //   Layer<float>::input1.resize(2);
     
    const  mxArray* mx_mat=prhs[1];
    const float* mat_mem_ptr = reinterpret_cast<const float*>(mxGetData(mx_mat));
 const mxArray *   mx_mat1=prhs[2];
    const float* channel = reinterpret_cast<const float*>(mxGetData(mx_mat1));
 const mxArray *  mx_mat2=prhs[3];
    const float* height = reinterpret_cast<const float*>(mxGetData(mx_mat2));
  const mxArray *   mx_mat3=prhs[4];
    const float* width = reinterpret_cast<const float*>(mxGetData(mx_mat3));


  //  Layer<float>::input1.resize(2);
   const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1]));
   LOG(INFO) << dims[3] << " " << dims[2] << " " << dims[1] << " " << dims[0];
  //  Blob<float> blob(1, 1, dims[1], dims[0]);
//   mx_mat_to_blob(prhs[1], &blob, DATA);
   Layer<float>::input1[0]->Reshape(1, 1, dims[1], dims[0]); 
//    Layer<float>::input1[0]->Reshape(1, 1, dims[1], dims[0]);
    mx_mat_to_blob(prhs[1],Layer<float>::input1[0], DATA);
  //   const float *data=blob.cpu_data();
 const float *    data=Layer<float>::input1[0]->cpu_data();
    printf("my_set_data1_ the data is %f\n\n\n",data[69258807]);
    
    
 }

 static void get_pad_feature(MEX_ARGS) {
   Blob<float>* tmp=new Blob<float>();
   tmp->Reshape(9,512,54,46); 
   caffe_copy(tmp->count(),Layer<float>::input1[0]->mutable_gpu_data(),tmp->mutable_gpu_data());
   plhs[0] = blob_to_mx_mat(tmp, DATA); 
    Blob<float>* tmp1=new Blob<float>();
   tmp1->Reshape(1,512,66,48); 
   caffe_copy(tmp1->count(),Layer<float>::padded_feature[0]->mutable_gpu_data(),tmp1->mutable_gpu_data());
   plhs[1] = blob_to_mx_mat(tmp1, DATA); 


 }


static void set_raw_feature(MEX_ARGS) {
   const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
   const int* dims2 = static_cast<const int*>(mxGetDimensions(prhs[2])); 
   Blob<float>* tmp=new Blob<float>();
   tmp->Reshape(1,dims1[2],dims1[1],dims1[0]);
   Blob<float>* tmp1=new Blob<float>();
   tmp1->Reshape(9,dims2[2],dims2[1],dims2[0]);
    Layer<float>::raw_feature.resize(1);
    Layer<float>::raw_feature[0].reset(new Blob<float>());
    Layer<float>::raw_feature[0]->Reshape(1,dims1[2],dims1[1],dims1[0]);
    Layer<float>::raw_feature_tmp.resize(1);
    Layer<float>::raw_feature_tmp[0].reset(new Blob<float>());
    Layer<float>::raw_feature_tmp[0]->Reshape(9,dims2[2],dims2[1],dims2[0]); 

    mx_mat_to_blob(prhs[1],tmp, DATA); 
    mx_mat_to_blob(prhs[2],tmp1, DATA);  

    switch (Caffe::mode()) {
            case Caffe::CPU: {
            caffe_copy(tmp->count(),tmp->mutable_cpu_data(),Layer<float>::raw_feature[0]->mutable_cpu_data());
            caffe_copy(tmp1->count(),tmp1->mutable_cpu_data(),Layer<float>::raw_feature_tmp[0]->mutable_cpu_data());
            }
            break;
            case Caffe::GPU: {
            caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::raw_feature[0]->mutable_gpu_data());
            caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::raw_feature_tmp[0]->mutable_gpu_data()); 
            }
            break;
            default:
            mxERROR("Unknown Caffe mode");
        }
    delete tmp; delete tmp1;
 }

// 我的一些定义
static void im2col(MEX_ARGS) {
    mxCHECK(nrhs == 7, "Usage: caffe_(im_data, kernel_h, kernel_w, stride_h, stride_w)");
    int num_dim = static_cast<int>(mxGetNumberOfDimensions(prhs[0]));
    mxCHECK(num_dim ==4, "Input data shoud have 4 dimensions.");
    const int* dims = static_cast<const int*>(mxGetDimensions(prhs[0])); 
    Blob<float> im_data(dims[3], dims[2], dims[1], dims[0]);
    LOG(INFO) << dims[3] << " " << dims[2] << " " << dims[1] << " " << dims[0];
  //  printf("we get here11\n\n\n");
   // sleep(10);

    mx_mat_to_blob(prhs[0], &im_data, DATA);
    LOG(INFO) << "Label 1";
    int kernel_h = static_cast<int>(mxGetScalar(prhs[1]));
    int kernel_w = static_cast<int>(mxGetScalar(prhs[2]));
    int stride_h = static_cast<int>(mxGetScalar(prhs[3]));
    int stride_w = static_cast<int>(mxGetScalar(prhs[4]));
    int pad_h = static_cast<int>(mxGetScalar(prhs[5]));
    int pad_w = static_cast<int>(mxGetScalar(prhs[6]));
    vector<int> out_shape;
    out_shape.push_back(dims[3]);
    out_shape.push_back(1);
    out_shape.push_back((dims[2]) * kernel_h * kernel_w);
    int out_h = (dims[1] - kernel_h+2*pad_h)/stride_h +1;
    int out_w = (dims[0] - kernel_w+2*pad_w)/stride_w +1;
    LOG(INFO) << "out_hxout_w: " << out_h << "x" << out_w; 
    out_shape.push_back(out_h * out_w);

    LOG(INFO) << out_shape[0] << " " << out_shape[1] << " " << out_shape[2] << " " << out_shape[3];
    Blob<float> col_data(out_shape);
    for (int n = 0; n < im_data.num(); n++) {
        switch (Caffe::mode()) {
            case Caffe::CPU: {
            const float* in_cpu = im_data.cpu_data() + im_data.offset(n);
            float* out_cpu = col_data.mutable_cpu_data() + col_data.offset(n);
            im2col_cpu(in_cpu, dims[2], dims[1], dims[0], kernel_h, kernel_w, pad_h, pad_w, 
                       stride_h, stride_w, out_cpu, 1, 1);
            }
            break;
            case Caffe::GPU: {
            const float* in_gpu = im_data.gpu_data() + im_data.offset(n);
            float* out_gpu = col_data.mutable_gpu_data() + col_data.offset(n);
            im2col_gpu(in_gpu, dims[2], dims[1], dims[0], kernel_h, kernel_w, pad_h, pad_w, 
                       stride_h, stride_w, out_gpu,1, 1);
            }
            break;
            default:
            mxERROR("Unknown Caffe mode");
        }
    }
    plhs[0] = blob_to_mx_mat(&col_data, DATA);
}






// Usage: caffe_('blob_get_diff', hBlob)
static void blob_get_diff(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('blob_get_diff', hBlob)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  plhs[0] = blob_to_mx_mat(blob, DIFF);
}

// Usage: caffe_('blob_set_diff', hBlob, new_diff)
static void blob_set_diff(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
      "Usage: caffe_('blob_set_diff', hBlob, new_diff)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  mx_mat_to_blob(prhs[1], blob, DIFF);
}

// Usage: caffe_('set_mode_cpu')
static void set_mode_cpu(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('set_mode_cpu')");
  Caffe::set_mode(Caffe::CPU);
}

// Usage: caffe_('set_mode_gpu')
static void set_mode_gpu(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('set_mode_gpu')");
  Caffe::set_mode(Caffe::GPU);
}

// Usage: caffe_('set_device', device_id)
static void set_device(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsDouble(prhs[0]),
      "Usage: caffe_('set_device', device_id)");
  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

// Usage: caffe_('get_init_key')
static void get_init_key(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('get_init_key')");
  plhs[0] = mxCreateDoubleScalar(init_key);
}

// Usage: caffe_('reset')
static void reset(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('reset')");
  // Clear solvers and stand-alone nets
  mexPrintf("Cleared %d solvers and %d stand-alone nets\n",
      solvers_.size(), nets_.size());
  solvers_.clear();
  nets_.clear();
  // Generate new init_key, so that handles created before becomes invalid
  init_key = static_cast<double>(caffe_rng_rand());
}

// Usage: caffe_('reset_by_index', index); index must be an descending order
static void delete_solver(MEX_ARGS) {
  mxCHECK(nrhs == 1, "Usage: caffe_('delete_solver', index)");
  int num_solver = mxGetNumberOfElements(prhs[0]);
  double* index = mxGetPr(prhs[0]);
  for (int i = 0; i < num_solver; i++) {
    LOG(INFO) << "solver size: " << solvers_.size();
    solvers_.erase(solvers_.begin() + static_cast<int>(index[i]) - 1);
  }

  // Clear solvers and stand-alone nets
  mexPrintf("Cleared %d solvers\n", num_solver);
  // Generate new init_key, so that handles created before becomes invalid
  // init_key = static_cast<double>(caffe_rng_rand());
}

// Usage: caffe_('read_mean', mean_proto_file)
static void read_mean(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsChar(prhs[0]),
      "Usage: caffe_('read_mean', mean_proto_file)");
  char* mean_proto_file = mxArrayToString(prhs[0]);
  mxCHECK_FILE_EXIST(mean_proto_file);
  Blob<float> data_mean;
  BlobProto blob_proto;
  bool result = ReadProtoFromBinaryFile(mean_proto_file, &blob_proto);
  mxCHECK(result, "Could not read your mean file");
  data_mean.FromProto(blob_proto);
  plhs[0] = blob_to_mx_mat(&data_mean, DATA);
  mxFree(mean_proto_file);
}

// Usage: caffe_('set_net_phase', hNet, phase_name)
static void set_net_phase(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('set_net_phase', hNet, phase_name)");
  char* phase_name = mxArrayToString(prhs[1]);
  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
    phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
    phase = TEST;
  } else {
    mxERROR("Unknown phase");
  }
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->set_net_phase(phase);
  mxFree(phase_name);
}

// Usage: caffe_('empty_net_param_diff', hNet)
static void empty_net_param_diff(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('empty_net_param_diff', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  for (int i = 0; i < net->params().size(); ++i) {
    shared_ptr<Blob<float> > blob = net->params()[i];
    switch (Caffe::mode()) {
      case Caffe::CPU:
	caffe_set(blob->count(), static_cast<float>(0),
	    blob->mutable_cpu_diff());
	break;
      case Caffe::GPU:
#ifndef CPU_ONLY
	caffe_gpu_set(blob->count(), static_cast<float>(0),
	    blob->mutable_gpu_diff());
#else
	NO_GPU;
#endif
	break;
    }
  }
}

// Usage: caffe_('apply_update', hSolver)
static void apply_update(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('apply_update', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  solver->MatCaffeApplyUpdate();
}

// Usage: caffe_('set_input_dim', hNet, dim)
static void set_input_dim(MEX_ARGS) {
  int blob_num = mxGetM(prhs[1]);
  int dim_num = mxGetN(prhs[1]);
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && dim_num == 5,
      "Usage: caffe_('set_input_dim', hNet, dim)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  double* dim = mxGetPr(prhs[1]);
  for (int blob_id = 0; blob_id < blob_num; blob_id++) {
    int id = static_cast<int>(dim[0*blob_num]);
    int n = static_cast<int>(dim[1*blob_num]);
    int c = static_cast<int>(dim[2*blob_num]);
    int h = static_cast<int>(dim[3*blob_num]);
    int w = static_cast<int>(dim[4*blob_num]);
    LOG(INFO) << "Reshape input blob.";
    LOG(INFO) << "Input_id = " << id;
    LOG(INFO) << "num = " << n;
    LOG(INFO) << "channel = " << c;
    LOG(INFO) << "height = " << h;
    LOG(INFO) << "width = " << w;
    net->input_blobs()[id]->Reshape(n, c, h, w);
    dim += 1;
  }
  // Reshape each layer of the network
  net->Reshape();
}

// Usage: caffe_('cnn2fcn', hNet)
static void cnn2fcn(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxGetNumberOfElements(prhs[1]) == 2, "Usage: caffe_('cnn2fcn', hNet, [kstride, pad])");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  double* param = mxGetPr(prhs[1]);
  int kstride = static_cast<int>(param[0]);
  int pad = static_cast<int>(param[1]);

  net->CNN2FCN(kstride, pad);
}

// Usage: caffe_('fcn2cnn', hNet)
static void fcn2cnn(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxGetNumberOfElements(prhs[1]) == 1,
      "Usage: caffe_('fcn2cnn', hNet, pad)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  double* pad_pr = mxGetPr(prhs[1]);
  int pad = static_cast<int>(pad_pr[0]);
  net->FCN2CNN(pad);
}

// Usage: caffe_('snapshot', hSolver, solver_name, model_name)
static void snapshot(MEX_ARGS) {
  mxCHECK(nrhs == 3 && mxIsStruct(prhs[0]),
      "Usage: caffe_('snapshot', hSolver, solver_name, model_name)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  string solver_name(mxArrayToString(prhs[1]));
  string model_name(mxArrayToString(prhs[2]));
  solver->MatCaffeSnapshot(solver_name, model_name);
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "get_solver",           get_solver           },
  { "solver_get_attr",      solver_get_attr      },
  { "solver_get_iter",      solver_get_iter      },
  { "solver_restore",       solver_restore       },
  { "solver_solve",         solver_solve         },
  { "solver_step",          solver_step          },
  { "get_net",              get_net              },
  { "net_get_attr",         net_get_attr         },
  { "net_forward",          net_forward          },
  { "net_backward",         net_backward         },
  { "net_copy_from",        net_copy_from        },
  { "net_reshape",          net_reshape          },
  { "net_save",             net_save             },
  { "layer_get_attr",       layer_get_attr       },
  { "layer_get_type",       layer_get_type       },
  { "blob_get_shape",       blob_get_shape       },
  { "blob_reshape",         blob_reshape         },
  { "blob_get_data",        blob_get_data        },
  { "blob_set_data",        blob_set_data        },
  { "my_set_data",          my_set_data          }, 
  { "my_set_data1",         my_set_data1         }, 
  { "my_set_data2",         my_set_data2         },
  { "my_set_data5",         my_set_data5         },
  { "my_set_data7",         my_set_data7         },
  { "my_set_data8",         my_set_data8         },
  { "my_set_data9",         my_set_data9         },
  { "set_raw_feature",      set_raw_feature      },
  { "get_pad_feature",      get_pad_feature      },
  { "set_momentum",         set_momentum         },
  { "im2col",               im2col               },
  { "blob_get_diff",        blob_get_diff        },
  { "blob_set_diff",        blob_set_diff        },
  { "set_mode_cpu",         set_mode_cpu         },
  { "set_mode_gpu",         set_mode_gpu         },
  { "set_device",           set_device           },
  { "get_init_key",         get_init_key         },
  { "reset",                reset                },
  { "delete_solver",        delete_solver        },
  { "read_mean",            read_mean            },
  { "set_net_phase",        set_net_phase        },    
  { "empty_net_param_diff", empty_net_param_diff },
  { "apply_update",         apply_update         },
  { "set_input_dim",        set_input_dim        },
  { "cnn2fcn",              cnn2fcn              },
  { "fcn2cnn",              fcn2cnn              },
  { "snapshot",             snapshot             },
  { "my_get_data",          my_get_data          },
  // The end.
  { "END",                  NULL                 },
};

/** -----------------------------------------------------------------
 ** matlab entry point.
 **/
// Usage: caffe_(api_command, arg1, arg2, ...)
void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  mxCHECK(nrhs > 0, "Usage: caffe_(api_command, arg1, arg2, ...)");
  // Handle input command
  char* cmd = mxArrayToString(prhs[0]);
  bool dispatched = false;
  // Dispatch to cmd handler
  for (int i = 0; handlers[i].func != NULL; i++) {
    if (handlers[i].cmd.compare(cmd) == 0) {
      handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
      dispatched = true;
      break;
    }
  }
  if (!dispatched) {
    ostringstream error_msg;
    error_msg << "Unknown command '" << cmd << "'";
    mxERROR(error_msg.str().c_str());
  }
  mxFree(cmd);
}
