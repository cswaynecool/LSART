#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void add_kernel1_cpu(const int n, const Dtype* a,
    const Dtype* b, const Dtype lambda1, const Dtype lambda2, Dtype* y) {
  for(int index=0;index<n;index++) {
    y[index] = a[index]*lambda1 + b[index]*lambda2;
  }
}

template <typename Dtype>
void WtfLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->ext_kernel_h_)
    / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->ext_kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void WtfLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   const Dtype* weight = this->blobs_[0]->cpu_data();
   
   const float * data=  Layer<float>::input[0]->cpu_data(); 

   const Dtype * parameter=Layer<Dtype>::input1[8]->cpu_data();
   Dtype lambda1=parameter[0]; Dtype lambda2=parameter[1];
   int conv_in_channels=data[0]; int conv_in_height=data[1]; int conv_in_width=data[2]; int height_in=data[3];
   int width_in= data[4]; int kernel_h=data[5]; int kernel_w=data[6];int pad_h=data[7];int pad_w=data[8]; 
   int stride_h=data[9]; int stride_w=data[10];

    int out_h = (conv_in_height - kernel_h+2*pad_h)/stride_h +1;
    int out_w = (conv_in_width - kernel_w+2*pad_w)/stride_w +1;

const Dtype* data1= bottom[0]->mutable_cpu_data();

          Dtype *col_buff=Layer<Dtype>::input[1]->mutable_cpu_data();
   
          Dtype* combined=Layer<Dtype>::input[16]->mutable_cpu_data();
 
         Dtype* weight_combined=Layer<Dtype>::input[14]->mutable_cpu_data();

    Dtype * combine=Layer<Dtype>::input1[7]->mutable_cpu_data();

add_kernel1_cpu(Layer<Dtype>::input1[7]->count(),data1,combine,lambda1,lambda2,combined);

add_kernel1_cpu(Layer<Dtype>::input1[9]->count(),this->blobs_[0]->mutable_cpu_data(),Layer<Dtype>::input1[9]->mutable_cpu_data(),lambda1,lambda2,weight_combined);

    Dtype* raw_feature=Layer<Dtype>::raw_feature[0]->mutable_cpu_data();
    Dtype* first_layer_col=Layer<Dtype>::first_layer_col[0]->mutable_cpu_data();
    im2col_cpu(raw_feature, 41, conv_in_height, conv_in_width,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, first_layer_col, 1, 1);

          Dtype* top_data1=Layer<Dtype>::first_layer_top_data[0]->mutable_cpu_data(); 
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 41*kernel_h*kernel_w,
        1,out_h*out_w,
        (Dtype)1., first_layer_col,weight_combined,
        (Dtype)0., top_data1); 


im2col_cpu(combined, conv_in_channels, conv_in_height, conv_in_width,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, col_buff, 1, 1);

    caffe_copy(Layer<Dtype>::input1[9]->count(),this->blobs_[0]->mutable_cpu_data(),Layer<Dtype>::input1[9]->mutable_cpu_data());

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data(); 

 caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_in_channels*kernel_h*kernel_w,
        1,out_h*out_w,
        (Dtype)1., col_buff,weight_combined,
        (Dtype)0., top_data); 
  }  
      
}

template <typename Dtype>
void WtfLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_cpu is not implemented for fully convolutin.";

const Dtype * data=  Layer<Dtype>::input[0]->cpu_data();
Dtype *bottom_data=bottom[0]->mutable_cpu_data();
const  int conv_in_channels_=data[0];const  int conv_in_height_=data[1];const int conv_in_width_=data[2];
const  int kernel_h_=data[5]; const int kernel_w_=data[6];const  int pad_h_=data[7]; const int pad_w_=data[8]; const int stride_h_=1;const int stride_w_=1; 

int out_h = (conv_in_height_ - kernel_h_+2*pad_h_)/stride_h_ +1;
    int out_w = (conv_in_width_ - kernel_w_+2*pad_w_)/stride_w_ +1;

  const Dtype* weight = this->blobs_[0]->cpu_data();
 Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
 
  Dtype *col_buff=Layer<Dtype>::input[1]->mutable_cpu_data();
  im2col_cpu(bottom_data, conv_in_channels_,conv_in_height_, conv_in_width_,
             kernel_h_, kernel_w_, pad_h_, pad_w_, 1, 1, col_buff, 1,1);

const Dtype *top_diff=top[0]->cpu_diff();
     
          Dtype * col_buff1=Layer<Dtype>::input[2]->mutable_cpu_data();

  im2col_cpu(top_diff, conv_in_channels_,kernel_h_, kernel_w_,
             kernel_h_, kernel_w_, 0, 0, 1, 1, col_buff1, 1,1); 
 
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,out_h*out_w,1, conv_in_channels_*kernel_h_*kernel_w_,
        (Dtype)1., col_buff,col_buff1,
        (Dtype)0., weight_diff);

    Dtype* raw_feature=Layer<Dtype>::raw_feature[0]->mutable_cpu_data();
    Dtype* first_layer_col=Layer<Dtype>::first_layer_col[0]->mutable_cpu_data();
    Dtype* second_layer_bottom_diff=Layer<Dtype>::second_layer_bottom_diff[0]->mutable_cpu_data();
    Dtype* first_layer_weight_diff=Layer<Dtype>::first_layer_weight_diff[0]->mutable_cpu_data();
    im2col_cpu(raw_feature, 41, conv_in_height_, conv_in_width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, first_layer_col, 1, 1);

    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,out_h*out_w,1, 41*kernel_h_*kernel_w_,
        (Dtype)1., first_layer_col,second_layer_bottom_diff,
        (Dtype)0., first_layer_weight_diff);
    caffe_add(this->blobs_[0]->count(), this->blobs_[0]->mutable_cpu_diff(), first_layer_weight_diff, this->blobs_[0]->mutable_cpu_diff());

}
#ifdef CPU_ONLY
STUB_GPU(WtfLayer);
#endif

INSTANTIATE_CLASS(WtfLayer);

}  // namespace caffe
