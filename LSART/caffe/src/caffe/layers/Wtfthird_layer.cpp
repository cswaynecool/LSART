#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {



template <typename Dtype>
void WtfthirdLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->ext_kernel_h_)
    / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->ext_kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void multipy_mask(const int count, Dtype* output, const unsigned int* mask) {
 for(int index=0;index<count;index++)
    {
      output[index]=output[index]*mask[index];
    }

}


template <typename Dtype>
void WtfthirdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    Dtype* weight = this->blobs_[0]->mutable_cpu_data();

  const  unsigned int* mask=this->mask_.cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

   multipy_mask(this->blobs_[0]->count(),weight,mask);  

     
    for (int n = 0; n < this->num_; ++n) {  
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }  
      
}

template <typename Dtype>
void WtfthirdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK((this->kstride_h_ == 1) && (this->kstride_w_ == 1)) << "Backward_cpu is not implemented for fully convolutin.";

const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
 
const unsigned int* mask = this->mask_.cpu_data();

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    Dtype* top_diff_mutable = top[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    // Mask
  //  if (this->has_mask_ && this->phase_ == TRAIN) {
  //    const unsigned int* mask = this->mask_.gpu_data();
  //    for (int n = 0; n < this->num_; ++n) {
  //  this->backward_gpu_mask(top_diff_mutable + top[i]->offset(n), mask);
  //    }
  //  }
   

    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
	  if (this->kstride_h_ == 1) {
	    this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
	  } else {
	    this->fcn_weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
	    //LOG(INFO) << "fcn_weight_gpu_gemm";
	  }
        }

     //this->backward_cpu_mask(weight_diff, mask); 
     multipy_mask(this->blobs_[0]->count(),weight_diff,mask);  
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
          
  //LOG(INFO) << "end of convolutionlayer backward_gpu";



}

#ifdef CPU_ONLY
STUB_GPU(WtfthirdLayer);
#endif

INSTANTIATE_CLASS(WtfthirdLayer);

}  // namespace caffe
