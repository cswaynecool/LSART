## Learning Spatial-Aware Regressions for Visual Tracking (LSART)

### Introduction
This package contains the source code to reproduce the experimental results of LSART. The source code is mainly written in MATLAB with a modifed Caffe framework.

### Usage

* Supported OS: the source code was tested on 64-bit Ubuntu 14.04 Linux OS, and it should also be executable in other linux distributions.

* Dependencies: 
 * A modified version of [caffe](http://caffe.berkeleyvision.org/) framework (included in the ./LSART/caffe folder) and all its dependencies.
 * Cuda enabled GPUs

* Installation: 
 1. Install caffe: we use a modified version of the original caffe framework. Compile the source code in the ./LSART/caffe directory and the matlab interface following the [installation instruction of caffe](http://caffe.berkeleyvision.org/installation.html). Compile the Matconvnet by running ./LSART/matconvnet/matlab/vl_compilenn.m.
 2. Download the 16-layer VGG network from https://gist.github.com/ksimonyan/211839e770f7b538e2d8, and put the caffemodel file under the ./LSART/model directory.
 3. Download the VGG-M network (imagenet-vgg-m-2048) from http://www.vlfeat.org/matconvnet/pretrained/, and put the mat file under the ./LSART/networks directory. Note that we do not use the VGG-M as the feature of our tracker. We need VGG-M model to make the feature extraction code of CCOT executable.
 4. Properly configure the code (see blow), and run ./vot-toolkit-master/wworkspace/run_experiments.m


### Settings

# 1. We provide both GPU (default) and CPU implementations of our codes, if you want to run our codes in the CPU mode, make changes as follows:
 (1) In ./LSART/set_tracker_param.m: change Line 1 to use_gpu=0;

# 2. To run our codes, the folling pathes should be adjusted based on where you put the codes (we use CODE_ROOT to denot the root directory)
 
 (1). In ./LSART/set_tracker_param.m: change Line 3 to "addpath('$CODE_ROOT/LSART/caffe/matlab/', '$CODE_ROOT/LSART/util');"
 (2). In ./LSART/set_tracker_param.m: change Line 5 to "addpath('$CODE_ROOT/LSART/BBR');"
 (3). In ./LSART/set_tracker_param.m: change Line 6 to "addpath(genpath('$CODE_ROOT/LSART/matconvnet'));"
 (4). In ./LSART/set_tracker_param.m: change Line 7 to "addpath(genpath('$CODE_ROOT/LSART/pdollar_toolbox'));"
 (5). In ./LSART/set_tracker_param.m: change Line 8 to "addpath('$CODE_ROOT/LSART/feature_extraction');"
 (6). In ./LSART/set_tracker_param.m: change Line 9 to "addpath(genpath('$CODE_ROOT/LSART/ccot_runfile'));"
 (7). In ./LSART/set_tracker_param.m: change Line 63 to "feature_solver_def_file = '$CODE_ROOT/LSART/model/feature_solver.prototxt';"
 (8). In ./LSART/set_tracker_param.m: change Line 64 to "model_file = '$CODE_ROOT/LSART/model/VGG_ILSVRC_16_layers.caffemodel';"
 (9). In ./LSART/set_tracker_param.m: change Line 68 to "feature_solver_def_file1 = 'CODE_ROOT/LSART/model/feature_solver1.prototxt';"
 (10). In ./LSART/set_tracker_param.m: change Line 69 to "model_file1 = '$CODE_ROOT/LSART/model/VGG_ILSVRC_16_layers.caffemodel';" 
 (11). In ./LSART/set_tracker_param.m: change Line 74 to "spn_solver_def_file = '$CODE_ROOT/LSART/model/spn_solver.prototxt'; "
 (12). In ./LSART/set_tracker_param.m: change Line 77 to "cnna_solver_def_file = '$CODE_ROOT/LSART/model/cnn-a_solver.prototxt';"
 (13). In ./LSART/set_tracker_param.m: change Line 80 to "cnnb_solver_def_file = '$CODE_ROOT/LSART/model/cnn-b_solver.prototxt';"
 (14). In ./LSART/set_tracker_param.m: change Line 83 to "cnn_my_solver_def_file = '$CODE_ROOT/LSART/model/cnn-my_solver.prototxt';"
 (15). In ./LSART/set_tracker_param.m: change Line 86 to "dtpooling_solver_def_file = '$CODE_ROOT/LSART/model/dtpooling_solver.prototxt';"
 (16). In ./LSART/set_tracker_param.m: change Line 89 to "KRR_solver_def_file = '$CODE_ROOT/LSART/model/cnn-KRR_solver.prototxt';"
 (17). In ./LSART/set_tracker_param.m: change Line 92 to "cnnc_solver_def_file = '$CODE_ROOT/LSART/model/cnn-c_solver.prototxt';"
 (18). In ./LSART/feature_extraction/get_table_feature.m: change Line 24 to "tables{end+1} = load(['$CODE_ROOT/LSART/feature_extraction/lookup_tables/'    fparam.tablename]);"
 (19). In ./LSART/feature_extraction/init_features.m: change Line 60 to "table = load(['$CODE_ROOT/LSART/feature_extraction/lookup_tables/' features{k}.fparams.tablename]);"
 (20). In ./LSART/feature_extraction/load_cnn.m: change Line 3 to "net = load(['$CODE_ROOT/LSART/networks/' fparams.nn_name]);"
 (21). In ./LSART/model/cnn-KRR_solver.prototxt: change Line 1 to net: '$CODE_ROOT/LSART/model/cnn-KRR.prototxt'
 (22). In ./LSART/model/cnn-a_solver.prototxt: change Line 1 to net: '$CODE_ROOT/LSART/model/cnn-a.prototxt'
 (23). In ./LSART/model/cnn-b_solver.prototxt: change Line 1 to net: '$CODE_ROOT/LSART/model/cnn-b.prototxt'
 (24). In ./LSART/model/cnn-c_solver.prototxt: change Line 1 to net: '$CODE_ROOT/LSART/model/cnn-c.prototxt'
 (25). In ./LSART/model/cnn-my_solver.prototxt: change Line 1 to net: '$CODE_ROOT/LSART/model/cnn-my.prototxt'
 (26). In ./LSART/model/dtpooling_solver.prototxt: change Line 1 to net: '$CODE_ROOT/LSART/model/dtpooling.prototxt'
 (27). In ./LSART/model/feature_solver.prototxt: change Line 1 to net: '$CODE_ROOT/LSART/model/feature_net.prototxt' 
 (28). In ./LSART/model/feature_solver1.prototxt: change Line 1 to net: '$CODE_ROOT/LSART/model/feature_net1.prototxt'
 (29). In ./LSART/model/spn_solver.prototxt: change Line 1 to net: '$CODE_ROOT/LSART/model/spn.prototxt'
We suggest the users to use tools (e.g., cscope) for bulk changes. 

### Contact
waynecool@mail.dlut.edu.cn


### Liscense

        Copyright (c) 2017, Chong Sun
        All rights reserved. 
        Redistribution and use in source and binary forms, with or without modification, are 
        permitted provided that the following conditions are met:
    		* Redistributions of source code must retain the above copyright 
      		  notice, this list of conditions and the following disclaimer.
    		* Redistributions in binary form must reproduce the above copyright 
      		  notice, this list of conditions and the following disclaimer in 
      		  the documentation and/or other materials provided with the distribution
        
        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
        ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 	
        LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
        CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
        INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
        CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
        ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
        POSSIBILITY OF SUCH DAMAGE.
