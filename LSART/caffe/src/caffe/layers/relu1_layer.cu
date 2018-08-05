#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void myReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
__global__ void pad_feature(const int n, const Dtype* feature, Dtype* padded_feature,int pad_h,int pad_w, int pad_feauture_h,
int pad_feature_w,int num_per_channel) {
  CUDA_KERNEL_LOOP(index, n) {
   int col=index%46; int row=index/46;
    int pad_row=row+pad_h;
    int pad_col=col+pad_w;
    int index_base=pad_row*pad_feature_w+pad_col;
    for (int channel_index=0; channel_index<512; channel_index++)
      { int current_index=index_base+channel_index*num_per_channel;
        padded_feature[current_index]=feature[index+46*46*channel_index];
          
      }


  }
}

template <typename Dtype>
__global__ void obtain_tmp(const int n, Dtype* padded_feature, Dtype* tmp, int pad_feature_h, int pad_feature_w, int kernel_h, int kernel_w,
int tmp_height,int tmp_width,int num_per_channel1,int num_per_channel2, int num_per_sample) {
  CUDA_KERNEL_LOOP(index, n) {

  int col=index%tmp_width; int row=index/tmp_width; 

      for(int row_index=0; row_index<3;row_index++)
       {  for(int col_index=0;col_index<3;col_index++ )
          {int layer_index=row_index*3+col_index;
           { int original_row=row+row_index*kernel_h/3; int original_col=col+col_index*kernel_w/3;
             int base_index=original_row*pad_feature_w+original_col;
               for(int channel_index=0;channel_index<512;channel_index++)
               {  int index1=base_index+channel_index*num_per_channel2;
                  int index2=index+num_per_channel1*channel_index+layer_index*num_per_sample;
                  tmp[index2]=padded_feature[index1]/2000;

               }
           }  
         }    
       }    
  }
}
    
template <typename Dtype>
__global__ void add_mask(const int n, Dtype* input_feature,Dtype* cos_win ,Dtype* output_feature) {
  CUDA_KERNEL_LOOP(index, n) {
     for(int channel_index=0;channel_index<512;channel_index++)
      { int index1=index+n*channel_index;
        output_feature[index1]=input_feature[index1]*cos_win[index];
          
      }

  }
 
}

template <typename Dtype>
void myReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  // NOLINT_NEXT_LINE(whitespace/operators)
  myReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
  Dtype* data=Layer<Dtype>::input[0]->mutable_cpu_data();
  int channels=data[0]; int kernel_h=data[5];int kernel_w=data[6]; int pad_h=data[7];int pad_w=data[8]; int height=46;
  int width=46;
  int pad_feature_h=height+2*pad_h; int pad_feature_w=width+2*pad_w;
  int num_per_channel=pad_feature_h*pad_feature_w;
  int count1=46*46;
  Dtype* cos_win=Layer<Dtype>::cos_window[0]->mutable_gpu_data(); 
  Dtype* tmp_feature=Layer<Dtype>::masked_feature[0]->mutable_gpu_data();
  add_mask<Dtype><<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,top_data, cos_win, tmp_feature);


  Dtype* padded_feature=Layer<Dtype>::padded_feature[0]->mutable_gpu_data();
        pad_feature<Dtype><<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1, tmp_feature, padded_feature,pad_h,pad_w, pad_feature_h,pad_feature_w,num_per_channel);

 int move_height= 46+2*pad_h -kernel_h +1;
 int move_width= 46+2*pad_w -kernel_w +1;
 int tmp_height=(move_height-1)+kernel_h/3;
 int tmp_width=(move_width-1)+kernel_w/3;
 int num_per_channel1=tmp_height*tmp_width;
 count1=tmp_height*tmp_width;
 Dtype* tmp=Layer<Dtype>::input1[0]->mutable_gpu_data();
 int num_per_sample=512*tmp_height*tmp_width;
 obtain_tmp<Dtype><<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(count1,padded_feature, tmp, pad_feature_h, pad_feature_w, kernel_h, kernel_w,
tmp_height,tmp_width,num_per_channel1,num_per_channel,num_per_sample);
}

template <typename Dtype>
__global__ void myReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
void myReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    myReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(myReLULayer);


}  // namespace caffe
