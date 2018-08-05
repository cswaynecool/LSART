use_gpu=1;
close all;
addpath('/media/wayne/h/LSART/LSART/caffe/matlab/', '/media/wayne/h/LSART/LSART/util');
addpath('./libsvm')
addpath('/media/wayne/h/LSART/LSART/BBR')
addpath(genpath('/media/wayne/h/LSART/LSART/matconvnet'))
addpath(genpath('/media/wayne/h/LSART/LSART/pdollar_toolbox'))
addpath('/media/wayne/h/LSART/LSART/feature_extraction');
addpath(genpath('/media/wayne/h/LSART/LSART/ccot_runfile'));
caffe.reset_all();

max_iter = 180;
mean_pix = [103.939, 116.779, 123.68]; 
seq.init_rect=init_rect;
%% Init location parameters
dia = (seq.init_rect(3)^2+seq.init_rect(4)^2)^0.5;
rec_scale_factor = [dia/seq.init_rect(3), dia/seq.init_rect(4)];
center_off = [0,0];
roi_scale = 2;
roi_scale_factor = roi_scale*[rec_scale_factor(1),rec_scale_factor(2)];

% img_sample_sz=floor([seq.init_rect(3)*roi_scale_factor(1) seq.init_rect(4)*roi_scale_factor(2)]);
img_sample_sz=[200 200];
width_in=floor(46/ roi_scale_factor(1)); height_in=floor( 46/ roi_scale_factor(2) );
kernel_h=max(3*floor(height_in/3),3); kernel_w=max(3*floor(width_in/3),3);  stride_w=1; stride_h=1; pad_h=floor(kernel_h/2); pad_w=floor(kernel_w/2);
% pad_h=0; pad_w=0;
if seq.init_rect(3)>30&&seq.init_rect(4)>60
   kernel_w=max(3*floor(width_in/3)-3,3); 
     kernel_h=max(3*floor(height_in/3)-3,3);
else
    kernel_w=max(3*ceil(width_in/3)-3,3); 
     kernel_h=max(3*ceil(height_in/3)-3,3);
end

     kernel_w1=kernel_w;
     kernel_h1=kernel_h;
  pad_h1=pad_h; pad_w1=pad_w;

seq.startFrame=1;

move_width=floor( (46+2*pad_w-kernel_w)/stride_w )+1;
move_height=floor( (46+2*pad_h -kernel_h)/stride_h )+1;

move_width1=floor( (46+2*pad_w1-kernel_w1)/stride_w )+1;
move_height1=floor( (46+2*pad_h1-kernel_h1)/stride_h )+1;
channel1=512;
width1=46;
height1=46;
stride_h1=1; stride_w1=1;

channel=kernel_h*kernel_w*512;
fid=fopen('data.txt','wt');
fprintf(fid,'%f\n%f\n%f\n%f\n%f\n%f\n%f',roi_scale_factor(1),roi_scale_factor(2),move_width,move_height,channel,kernel_h,kernel_w);
fclose(fid);

fid=fopen('data1.txt','wt');
fprintf(fid,'%f\n%f\n%f\n%f',move_width1,move_height1,kernel_h1,kernel_w1);
fclose(fid);
%% init caffe %%
if use_gpu==1
    gpu_id = 0;
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();   
end
feature_solver_def_file = '/media/wayne/h/LSART/LSART/model/feature_solver.prototxt';
model_file = '/media/wayne/h/LSART/LSART/model/VGG_ILSVRC_16_layers.caffemodel';
fsolver = caffe.Solver(feature_solver_def_file);
fsolver.net.copy_from(model_file);

feature_solver_def_file1 = '/media/wayne/h/LSART/LSART/model/feature_solver1.prototxt';
model_file1 = '/media/wayne/h/LSART/LSART/model/VGG_ILSVRC_16_layers.caffemodel';
fsolver1 = caffe.Solver(feature_solver_def_file1);
fsolver1.net.copy_from(model_file1);
%% spn solver

spn_solver_def_file = '/media/wayne/h/LSART/LSART/model/spn_solver.prototxt';
spn = caffe.Solver(spn_solver_def_file);
%% cnn-a solver
cnna_solver_def_file = '/media/wayne/h/LSART/LSART/model/cnn-a_solver.prototxt'; 
cnna = caffe.Solver(cnna_solver_def_file);

cnnb_solver_def_file = '/media/wayne/h/LSART/LSART/model/cnn-b_solver.prototxt'; 
cnnb = caffe.Solver(cnnb_solver_def_file);

 cnn_my_solver_def_file = '/media/wayne/h/LSART/LSART/model/cnn-my_solver.prototxt'; 
 cnn_my = caffe.Solver(cnn_my_solver_def_file);

  dtpooling_solver_def_file = '/media/wayne/h/LSART/LSART/model/dtpooling_solver.prototxt'; 
 cnn_dtpooling = caffe.Solver(dtpooling_solver_def_file);
 
 KRR_solver_def_file = '/media/wayne/h/LSART/LSART/model/cnn-KRR_solver.prototxt'; 
 KRR = caffe.Solver(KRR_solver_def_file);
 
  cnnc_solver_def_file = '/media/wayne/h/LSART/LSART/model/cnn-c_solver.prototxt'; 
  cnnc = caffe.Solver(cnnc_solver_def_file);
 

map_sigma_factor = 1/12;
roi_size = 361;
location = seq.init_rect;
%% init ensemble parameters
ensemble_num = 100;
w0 = single(ones(1, 1, ensemble_num, 1));
wt0 = w0;
wt = single(zeros(1, 1, ensemble_num, 1));
eta = 0.2; % weight of selected feature maps
%% Init scale parameters
scale_param = init_scale_estimator;
%%
params=testing();
params.wsize = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
pos = floor(params.init_pos(:)');
target_sz = floor(params.wsize(:)');
init_target_sz = target_sz;
search_area_scale = params.search_area_scale;
features = params.t_features;

prior_weights = [];
sample_weights = [];
latest_ind = [];

if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
if ~isfield(params, 'interpolation_method')
    params.interpolation_method = 'none';
end
if ~isfield(params, 'interpolation_centering')
    params.interpolation_centering = false;
end
if ~isfield(params, 'interpolation_windowing')
    params.interpolation_windowing = false;
end
if ~isfield(params, 'clamp_position')
    params.clamp_position = false;
end
im_ccot=im1;
if size(im1,3) == 3
    if all(all(im1(:,:,1) == im1(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im1,3) > 1 && is_color_image == false
    im_ccot = im1(:,:,1);
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');
img_sample_sz = feature_info.img_sample_sz;
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
feature_dim = feature_info.dim;
num_feature_blocks = length(feature_dim);
a=1;
