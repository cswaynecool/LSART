#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void WtfsecondLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
      //  printf("this values is %d-> %d-> %d-> %d\n\n",this->blobs_[0]->num(), this->blobs_[0]->channels(),this->blobs_[0]->height(),this->blobs_[0]->width());
         // sleep(10);
  const Dtype * data=  Layer<Dtype>::input[0]->cpu_data(); 
  int conv_in_channels_=data[0]; int conv_in_height_=data[5];int conv_in_width_=data[6];
  int kernel_h_=conv_in_height_; int kernel_w_=conv_in_width_; int pad_h_=0; int pad_w_=0; int stride_h_=1; int stride_w_=1;
   const Dtype * parameter=Layer<Dtype>::input1[8]->cpu_data();
   Dtype lambda1=parameter[0]; Dtype lambda2=parameter[1];
//printf("the second layer\n\n");

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype *data_im=Layer<Dtype>::input[6]->mutable_gpu_data();
     

   col2im_gpu(bottom_data, conv_in_channels_,
    kernel_h_, kernel_w_,kernel_h_, kernel_w_,0,0,
    1,1,data_im);      

    Dtype* top_data = top[0]->mutable_gpu_data();

        Dtype* template1=Layer<Dtype>::input[4]->mutable_gpu_data();
im2col_gpu(data_im, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, kernel_h_/3, kernel_w_/3, template1, 1, 1);

    int out_h = (Layer<Dtype>::input1[0]->height() - kernel_h_/3+2*pad_h_)/stride_h_ +1;
    int out_w = (Layer<Dtype>::input1[0]->width() - kernel_w_/3+2*pad_w_)/stride_w_ +1;
   
    Dtype* tmp1=Layer<Dtype>::input[7]->mutable_gpu_data();  

    Dtype * col_buff=Layer<Dtype>::input[8]->mutable_gpu_data();

Dtype *  data1= Layer<Dtype>::input1[0]->mutable_gpu_data();

    Dtype* weight_combined=Layer<Dtype>::input[15]->mutable_gpu_data();
    caffe_gpu_add1(Layer<Dtype>::input1[10]->count(),this->blobs_[0]->mutable_gpu_data(),Layer<Dtype>::input1[10]->mutable_gpu_data(),lambda1,lambda2,weight_combined);   
    
     for (int ii=0;ii<3;ii++)
          for (int jj=0;jj<3;jj++)
        { 
            int layer_index=3*ii+jj;  
            im2col_gpu(data1+Layer<Dtype>::input1[0]->offset(layer_index), conv_in_channels_,Layer<Dtype>::input1[0]->height() , Layer<Dtype>::input1[0]->width(),
             kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, 1, 1, col_buff, 1,1);

            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,9 , out_h*out_w, conv_in_channels_*kernel_h_/3*kernel_w_/3,
            (Dtype)1., template1 , col_buff,
            (Dtype)0, tmp1);

            if (ii==0&&jj==0)
            {
           caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,1 , out_h*out_w, 9,
           (Dtype)1., weight_combined +layer_index*9 , tmp1,
           (Dtype)0, top_data);
           }
            else
            {
               caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,1 , out_h*out_w, 9,
               (Dtype)1., weight_combined +layer_index*9 , tmp1,
               (Dtype)1, top_data);
            }
           
        } 
 Dtype* bottom_data1=Layer<Dtype>::first_layer_top_data[0]->mutable_gpu_data();
 Dtype* second_layer_template1=Layer<Dtype>::second_layer_template1[0]->mutable_gpu_data();
 Dtype* second_layer_data1=Layer<Dtype>::raw_feature_tmp[0]->mutable_gpu_data();
 Dtype* second_layer_col_buff=Layer<Dtype>::second_layer_col_buff[0]->mutable_gpu_data();
 Dtype* second_layer_tmp1=Layer<Dtype>::second_layer_tmp1[0]->mutable_gpu_data();
 Dtype* second_layer_top_data=Layer<Dtype>::second_layer_top_data[0]->mutable_gpu_data();


 im2col_gpu(bottom_data1, 41, conv_in_height_, conv_in_width_,
        kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, kernel_h_/3, kernel_w_/3, second_layer_template1, 1, 1);

for (int ii=0;ii<3;ii++)
          for (int jj=0;jj<3;jj++)
          {
             int layer_index=3*ii+jj; 
            im2col_gpu(second_layer_data1+Layer<Dtype>::raw_feature_tmp[0]->offset(layer_index), 41, Layer<Dtype>::raw_feature_tmp[0]->height() , Layer<Dtype>::raw_feature_tmp[0]->width(),
             kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, 1, 1, second_layer_col_buff, 1,1);

             caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,9 , out_h*out_w, 41*kernel_h_/3*kernel_w_/3,
            (Dtype)1., second_layer_template1, second_layer_col_buff,
            (Dtype)0, second_layer_tmp1);

           if (ii==0&&jj==0)
            {
           caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,1 , out_h*out_w, 9,
           (Dtype)1., weight_combined +layer_index*9 , second_layer_tmp1,
           (Dtype)0, second_layer_top_data);
           }
            else
            {
               caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,1 , out_h*out_w, 9,
               (Dtype)1., weight_combined +layer_index*9 , second_layer_tmp1,
               (Dtype)1, second_layer_top_data);
            }
              

          }

 caffe_gpu_add(top[0]->count(), top_data, second_layer_top_data,top_data);
caffe_copy(Layer<Dtype>::input1[10]->count(),this->blobs_[0]->mutable_gpu_data(),Layer<Dtype>::input1[10]->mutable_gpu_data());

}

template <typename Dtype>
void WtfsecondLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//  LOG(INFO) << "start of convolutionlayer backward_gpu";
  //        sleep(10);
  //CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_gpu is not implemented for fully convolutin.";
   Dtype *  data1= Layer<Dtype>::input1[0]->mutable_gpu_data(); 
  const Dtype* weight = this->blobs_[0]->gpu_data();
 Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff(); 
 const Dtype* top_diff=top[0]->gpu_diff();
Dtype * bottom_diff=bottom[0]->mutable_gpu_diff();
const Dtype* bottom_data = bottom[0]->gpu_data();
 const Dtype * data=  Layer<Dtype>::input[0]->cpu_data(); 
  int conv_in_channels_=data[0]; int conv_in_height_=data[5];int conv_in_width_=data[6];
  int kernel_h_=conv_in_height_; int kernel_w_=conv_in_width_; int pad_h_=0; int pad_w_=0; int stride_h_=1; int stride_w_=1; 
   
    Dtype* top_data = top[0]->mutable_gpu_data();

          Dtype *template1=Layer<Dtype>::input[9]->mutable_gpu_data();

          Dtype* data_im=Layer<Dtype>::input1[4]->mutable_gpu_data();
col2im_gpu(bottom_data, conv_in_channels_,
    conv_in_height_, conv_in_width_, kernel_h_, kernel_w_,
    0, 0,  stride_h_,
    stride_w_,data_im);


im2col_gpu(data_im, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, kernel_h_/3, kernel_w_/3, template1, 1, 1);

    int out_h = (Layer<Dtype>::input1[0]->height() - kernel_h_/3+2*pad_h_)/stride_h_ +1;
    int out_w = (Layer<Dtype>::input1[0]->width() - kernel_w_/3+2*pad_w_)/stride_w_ +1;

   Dtype*  tmp1=Layer<Dtype>::input[10]->mutable_gpu_data();

    Dtype *col_buff=Layer<Dtype>::input[11]->mutable_gpu_data();

    Dtype* tmp3=Layer<Dtype>::input[12]->mutable_gpu_data();

    Dtype* tmp4=Layer<Dtype>::input[13]->mutable_gpu_data();

 for (int ii=0;ii<3;ii++)
          for (int jj=0;jj<3;jj++)
          {//
             int layer_index=3*ii+jj;  
             Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

          im2col_gpu(data1+Layer<Dtype>::input1[0]->offset(layer_index), conv_in_channels_,Layer<Dtype>::input1[0]->height() , Layer<Dtype>::input1[0]->width(),
             kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, 1, 1, col_buff, 1,1);
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,9 , out_h*out_w, conv_in_channels_*kernel_h_/3*kernel_w_/3,
            (Dtype)1., template1 , col_buff,
            (Dtype)0, tmp1);
           caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,9 , 1, out_h*out_w,
           (Dtype)1., tmp1, top_diff,
           (Dtype)0, weight_diff+layer_index*9); 
     
           caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,out_h*out_w , 9, 1,
           (Dtype)1., top_diff,weight+layer_index*9,
           (Dtype)0, tmp3); 
  
          if(ii==0&&jj==0)
           {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,conv_in_channels_*kernel_h_/3*kernel_w_/3 , 9,out_h*out_w,
           (Dtype)1., col_buff,tmp3,
           (Dtype)0, tmp4);}
           else
           {
             caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,conv_in_channels_*kernel_h_/3*kernel_w_/3 , 9,out_h*out_w,
             (Dtype)1., col_buff,tmp3,
             (Dtype)1, tmp4); 
           }
        }


col2im_gpu(tmp4, conv_in_channels_,
    kernel_h_, kernel_w_,kernel_h_/3 , kernel_w_/3,0,0,
    kernel_h_/3, kernel_w_/3,Layer<Dtype>::input1[6]->mutable_gpu_data());      

im2col_gpu(Layer<Dtype>::input1[6]->mutable_gpu_data(), conv_in_channels_,kernel_h_, kernel_w_,
             kernel_h_, kernel_w_, pad_h_, pad_w_, 1, 1, bottom_diff, 1,1);

 Dtype* bottom_data1=Layer<Dtype>::first_layer_top_data[0]->mutable_gpu_data();
 Dtype* second_layer_template1=Layer<Dtype>::second_layer_template1[0]->mutable_gpu_data();
 Dtype* second_layer_data1=Layer<Dtype>::raw_feature_tmp[0]->mutable_gpu_data();
 Dtype* second_layer_col_buff=Layer<Dtype>::second_layer_col_buff[0]->mutable_gpu_data();
 Dtype* second_layer_tmp1=Layer<Dtype>::second_layer_tmp1[0]->mutable_gpu_data();
 Dtype* second_layer_top_data=Layer<Dtype>::second_layer_top_data[0]->mutable_gpu_data();
 Dtype* second_layer_weight_diff=Layer<Dtype>::second_layer_weight_diff[0]->mutable_gpu_data();
 Dtype* second_layer_tmp3=Layer<Dtype>::second_layer_tmp3[0]->mutable_gpu_data();
 Dtype* second_layer_tmp4=Layer<Dtype>::second_layer_tmp4[0]->mutable_gpu_data();
im2col_gpu(bottom_data1, 41, conv_in_height_, conv_in_width_,
        kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, kernel_h_/3, kernel_w_/3, second_layer_template1, 1, 1);
 for (int ii=0;ii<3;ii++)
          for (int jj=0;jj<3;jj++)
          {
              
              {
                   int layer_index=3*ii+jj;  
                   im2col_gpu(second_layer_data1+Layer<Dtype>::raw_feature_tmp[0]->offset(layer_index), 41, Layer<Dtype>::raw_feature_tmp[0]->height() , Layer<Dtype>::raw_feature_tmp[0]->width(),
                   kernel_h_/3, kernel_w_/3, pad_h_, pad_w_, 1, 1, second_layer_col_buff, 1,1);

                 caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,9 , out_h*out_w, 41*kernel_h_/3*kernel_w_/3,
                 (Dtype)1., second_layer_template1, second_layer_col_buff,
                 (Dtype)0, second_layer_tmp1);
           
                 caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,9 , 1, out_h*out_w,
                 (Dtype)1., second_layer_tmp1, top_diff,
                 (Dtype)0, second_layer_weight_diff+layer_index*9); 

                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,out_h*out_w , 9, 1,
                (Dtype)1., top_diff,weight+layer_index*9,
                (Dtype)0, second_layer_tmp3);

                 if(ii==0&&jj==0)
                 {
                  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,41*kernel_h_/3*kernel_w_/3 , 9,out_h*out_w,
                  (Dtype)1., second_layer_col_buff,second_layer_tmp3,
                  (Dtype)0, second_layer_tmp4);
                 }
                else
                {
                 caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,41*kernel_h_/3*kernel_w_/3 , 9,out_h*out_w,
                 (Dtype)1., second_layer_col_buff,second_layer_tmp3,
                 (Dtype)1, second_layer_tmp4); 
                }
              } 
          }

Dtype* second_layer_bottom_diff=Layer<Dtype>::second_layer_bottom_diff[0]->mutable_gpu_data();
col2im_gpu(second_layer_tmp4, 41,
    kernel_h_, kernel_w_,kernel_h_/3 , kernel_w_/3,0,0,
    kernel_h_/3, kernel_w_/3,second_layer_bottom_diff);      

caffe_gpu_add(this->blobs_[0]->count(), this->blobs_[0]->mutable_gpu_diff(), second_layer_weight_diff, this->blobs_[0]->mutable_gpu_diff());


  //LOG(INFO) << "end of convolutionlayer backward_gpu";
}

INSTANTIATE_LAYER_GPU_FUNCS(WtfsecondLayer);

}  // namespace caffe
