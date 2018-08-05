#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "iostream"
namespace caffe {




template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  CHECK((!conv_param.has_kstride() && conv_param.has_kstride_h() 
       && conv_param.has_kstride_w())
      || (!conv_param.has_kstride_h() && !conv_param.has_kstride_w()))
      << "kstride is kstride OR kstride_h and kstride_w are required.";

  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  if (!conv_param.has_kstride_h()) {
    kstride_h_ = kstride_w_ = conv_param.kstride();
  } else {
    kstride_h_ = conv_param.kstride_h();
    kstride_w_ = conv_param.kstride_w();
  }
  if (!(kstride_h_ == 1)) {
    CHECK_EQ(stride_h_, 1) << "Currently, when using kstride, the stride parameter should be fixed to 1.";
    CHECK_EQ(stride_w_, 1) << "Currently, when using kstride, the stride parameter should be fixed to 1.";
  }
  ext_kernel_h_ = (kernel_h_ - 1) * kstride_h_ + 1;
  ext_kernel_w_ = (kernel_w_ - 1) * kstride_w_ + 1;
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";

FILE *r=fopen("./data.txt","r");
        float data[7];
        fscanf(r,"%f %f %f %f %f %f %f",&data[0],&data[1],&data[2],&data[3],&data[4],&data[5],&data[6]);
        fclose(r);
FILE *r1=fopen("./data1.txt","r");
        float data1[7];
        fscanf(r1,"%f %f %f %f %f %f %f",&data1[0],&data1[1],&data1[2],&data1[3]);
        fclose(r1);
        int width_in=46/data[0]; int height_in=46/data[1];
        int move_width=data[2]; int move_height=data[3];
        int move_width1=data1[0]; int move_height1=data1[1];


  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  } // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    //
    //
     if (this->layer_param_.convolution_param().mylayerone()==1)
     {
        
        this->blobs_[0].reset(new Blob<Dtype>(
        1,1,move_width*move_height,1));

       
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

     }

    else if (this->layer_param_.convolution_param().mylayerseventh()==1)
     {
        
        this->blobs_[0].reset(new Blob<Dtype>(
        1,1,13,4));
    
       
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    Dtype* weight_cpu=this->blobs_[0]->mutable_cpu_data();
   // weight_cpu[0]=0; weight_cpu[1]=-1; weight_cpu[2]=0; weight_cpu[3]=-1;
     }

   else if(this->layer_param_.convolution_param().mylayertwo()==1)
   {
     this->blobs_[0].reset(new Blob<Dtype>(
        1,1,1,81));
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    Dtype *weight=this->blobs_[0]->mutable_cpu_data();
  for(int i=0;i<81;i++)
    weight[i]=weight[i]*10000;
   weight[40]=weight[40]+1; weight[10]=weight[10]+1; weight[20]=weight[20]+1; weight[0]=weight[0]+1; weight[30]=weight[30]+1; weight[50]=weight[50]+1; weight[60]=weight[60]+1;
    weight[70]=weight[70]+1; weight[80]=weight[80]+1;
   }
   else if(this->layer_param_.convolution_param().mylayerninth()==1)
   {
     this->blobs_[0].reset(new Blob<Dtype>(
        1,1,9,4));
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
   }
 

    else if (this->layer_param_.convolution_param().mylayerthird()==1)
    { 
        int out_h_tmp = (48 - 5+2*2)/1 +1;
        int out_w_tmp = (48 - 5+2*2)/1 +1; 
        this->blobs_[0].reset(new Blob<Dtype>(1,1,20,out_h_tmp*out_w_tmp));
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
        weight_filler->Fill(this->blobs_[0].get());
    }
    else if (this->layer_param_.convolution_param().mylayersixth()==1)
    { 
      this->blobs_[0].reset(new Blob<Dtype>(1,1,move_width1*move_height1,1));
     shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    }
   else
   {   
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
   }
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // Initialize mask_ if necessary
  has_mask_ = conv_param.mask();

  if (this->layer_param_.convolution_param().mylayerseventh()==1)
  {
   dt_max_idx_.Reshape(1,25,46,46); 
  }

   if (this->layer_param_.convolution_param().mylayerninth()==1)
  {
   ninthlayer_max_idx_.Reshape(1,9,move_height-16,move_width-16);
    printf("in initialization height and width are %d %d\n\n",move_height-16,move_width-16);
  }


  if (has_mask_) {
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    compute_output_shape();
    mask_.Reshape(52, conv_in_channels_, 5, 5);
    mask_index_.Reshape(1, 52, 25, 1);
    caffe_rng_bernoulli(mask_index_.count(), 0.3, mask_index_.mutable_cpu_data());
    fill_mask();
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::fill_mask_kernel_cpu(const int n, const unsigned int* mask_index,
    const int channel, const int height, const int width, 
    const int half_height, const int half_width, const int dim,
    unsigned int* mask) {
  for(int index=0;index<n;index++) {
   const int c1= index / dim;
   const int c2=c1/channel;
   const int c = index / dim;
   const int h = (index - c * dim) / width;
   const int w = index % width;
   const int h_ind = h / half_height;
   const int w_ind = w / half_width;
   mask[index] = mask_index[c2 * 25 + 5 * h_ind + w_ind];
  }
}


template <typename Dtype>
void BaseConvolutionLayer<Dtype>::fill_mask_cpu() {

  const unsigned int* mask_index = this->mask_index_.cpu_data();
  unsigned int* mask = this->mask_.mutable_cpu_data();
  const int channel = this->mask_.channels();
  const int height = this->mask_.height();
  const int half_height = height / 5;
  const int width = this->mask_.width();
  const int half_width = width / 5;
  const int dim = height * width;
  const int count = this->mask_.count();
  fill_mask_kernel_cpu(count, mask_index, channel, height, width, half_height, half_width, dim, mask);

}
  
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::fill_mask() {
  // Fix mask_index of the first five channels
  unsigned int* mask_index = this->mask_index_.mutable_cpu_data(); 
  CHECK_GT(this->mask_index_.channels(), 4) << "Channel number of mask_ (top blob) should be >= 5";
  CHECK_GE(this->mask_.width(), 2) << "width of mask_ (top blob) should be >= 2";
  CHECK_GE(this->mask_.height(), 2) << "height of mask_ (top blob) should be >= 2";
 
  mask_index[0] = 1;
  mask_index[1] = 1;
  mask_index[2] = 1;
  mask_index[3] = 1;
  mask_index[4] = 1;
  mask_index[5] = 1;
  mask_index[6] = 1;
  mask_index[7] = 1;
  mask_index[8] = 1; 
  mask_index[9] = 1;
  mask_index[10] = 0;
  mask_index[11] = 0;
  mask_index[12] = 0;
  mask_index[13] = 0;
  mask_index[14] = 0; 
  mask_index[15] = 0;
  mask_index[16] = 0;
  mask_index[17] = 0;
  mask_index[18] = 0;
  mask_index[19] = 0;
  mask_index[20] = 0;
  mask_index[21] = 0;
  mask_index[22] = 0;
  mask_index[23] = 0;
  mask_index[24] = 0;
int base=24;
  mask_index[1+base] = 0;
  mask_index[2+base] = 0; 
  mask_index[3+base] = 0;
  mask_index[4+base] = 0;
  mask_index[5+base] = 0;
  mask_index[6+base] = 1;
  mask_index[7+base] = 1;
  mask_index[8+base] = 1;
  mask_index[9+base] = 1;
  mask_index[10+base] = 1;
  mask_index[11+base] = 1;
  mask_index[12+base] = 1;
  mask_index[13+base] = 1;
  mask_index[14+base] = 1;
  mask_index[15+base] = 1;
  mask_index[16+base] = 0;
  mask_index[17+base] = 0;
  mask_index[18+base] = 0;
  mask_index[19+base] = 0;
  mask_index[20+base] = 0;
  mask_index[21+base] = 0;
  mask_index[22+base] = 0;
  mask_index[23+base] = 0;
  mask_index[24+base] = 0;
  mask_index[25+base] = 0;
base=49;
  mask_index[1+base] = 0;
  mask_index[2+base] = 0; 
  mask_index[3+base] = 0;
  mask_index[4+base] = 0;
  mask_index[5+base] = 0;
  mask_index[6+base] = 0;
  mask_index[7+base] = 0;
  mask_index[8+base] = 0;
  mask_index[9+base] = 0;
  mask_index[10+base] = 0;
  mask_index[11+base] = 1;
  mask_index[12+base] = 1;
  mask_index[13+base] = 1;
  mask_index[14+base] = 1;
  mask_index[15+base] = 1;
  mask_index[16+base] = 1;
  mask_index[17+base] = 1;
  mask_index[18+base] = 1;
  mask_index[19+base] = 1;
  mask_index[20+base] = 1;
  mask_index[21+base] = 0;
  mask_index[22+base] = 0;
  mask_index[23+base] = 0;
  mask_index[24+base] = 0;
  mask_index[25+base] = 0;
base=74;
  mask_index[1+base] = 0;
  mask_index[2+base] = 0; 
  mask_index[3+base] = 0;
  mask_index[4+base] = 0;
  mask_index[5+base] = 0;
  mask_index[6+base] = 0;
  mask_index[7+base] = 0;
  mask_index[8+base] = 0;
  mask_index[9+base] = 0;
  mask_index[10+base] = 0;
  mask_index[11+base] = 0;
  mask_index[12+base] = 0;
  mask_index[13+base] = 0;
  mask_index[14+base] = 0;
  mask_index[15+base] = 0;
  mask_index[16+base] = 1;
  mask_index[17+base] = 1;
  mask_index[18+base] = 1;
  mask_index[19+base] = 1;
  mask_index[20+base] = 1;
  mask_index[21+base] = 1;
  mask_index[22+base] = 1;
  mask_index[23+base] = 1;
  mask_index[24+base] = 1;
  mask_index[25+base] = 1;
base=99;
  mask_index[1+base] = 1;
  mask_index[2+base] = 1; 
  mask_index[3+base] = 0;
  mask_index[4+base] = 0;
  mask_index[5+base] = 0;
  mask_index[6+base] = 1;
  mask_index[7+base] = 1;
  mask_index[8+base] = 0;
  mask_index[9+base] = 0;
  mask_index[10+base] = 0;
  mask_index[11+base] = 1;
  mask_index[12+base] = 1;
  mask_index[13+base] = 0;
  mask_index[14+base] = 0;
  mask_index[15+base] = 0;
  mask_index[16+base] = 1;
  mask_index[17+base] = 1;
  mask_index[18+base] = 0;
  mask_index[19+base] = 0;
  mask_index[20+base] = 0;
  mask_index[21+base] = 1;
  mask_index[22+base] = 1;
  mask_index[23+base] = 0;
  mask_index[24+base] = 0;
  mask_index[25+base] = 0;
base=124;
  mask_index[1+base] = 0;
  mask_index[2+base] = 1; 
  mask_index[3+base] = 1;
  mask_index[4+base] = 0;
  mask_index[5+base] = 0;
  mask_index[6+base] = 0;
  mask_index[7+base] = 1;
  mask_index[8+base] = 1;
  mask_index[9+base] = 0;
  mask_index[10+base] = 0;
  mask_index[11+base] = 0;
  mask_index[12+base] = 1;
  mask_index[13+base] = 1;
  mask_index[14+base] = 0;
  mask_index[15+base] = 0;
  mask_index[16+base] = 0;
  mask_index[17+base] = 1;
  mask_index[18+base] = 1;
  mask_index[19+base] = 0;
  mask_index[20+base] = 0;
  mask_index[21+base] = 0;
  mask_index[22+base] = 1;
  mask_index[23+base] = 1;
  mask_index[24+base] = 0;
  mask_index[25+base] = 0;
base=149;
  mask_index[1+base] = 0;
  mask_index[2+base] = 0; 
  mask_index[3+base] = 1;
  mask_index[4+base] = 1;
  mask_index[5+base] = 0;
  mask_index[6+base] = 0;
  mask_index[7+base] = 0;
  mask_index[8+base] = 1;
  mask_index[9+base] = 1;
  mask_index[10+base] = 0;
  mask_index[11+base] = 0;
  mask_index[12+base] = 0;
  mask_index[13+base] = 1;
  mask_index[14+base] = 1;
  mask_index[15+base] = 0;
  mask_index[16+base] = 0;
  mask_index[17+base] = 0;
  mask_index[18+base] = 1;
  mask_index[19+base] = 1;
  mask_index[20+base] = 0;
  mask_index[21+base] = 0;
  mask_index[22+base] = 0;
  mask_index[23+base] = 1;
  mask_index[24+base] = 1;
  mask_index[25+base] = 0;
base=174;
  mask_index[1+base] = 0;
  mask_index[2+base] = 0; 
  mask_index[3+base] = 0;
  mask_index[4+base] = 1;
  mask_index[5+base] = 1;
  mask_index[6+base] = 0;
  mask_index[7+base] = 0;
  mask_index[8+base] = 0;
  mask_index[9+base] = 1;
  mask_index[10+base] = 1;
  mask_index[11+base] = 0;
  mask_index[12+base] = 0;
  mask_index[13+base] = 0;
  mask_index[14+base] = 1;
  mask_index[15+base] = 1;
  mask_index[16+base] = 0;
  mask_index[17+base] = 0;
  mask_index[18+base] = 0;
  mask_index[19+base] = 1;
  mask_index[20+base] = 1;
  mask_index[21+base] = 0;
  mask_index[22+base] = 0;
  mask_index[23+base] = 0;
  mask_index[24+base] = 1;
  mask_index[25+base] = 1;
  
  switch (Caffe::mode()) {
  case Caffe::CPU:
    fill_mask_cpu();
    break;
  case Caffe::GPU:
    fill_mask_gpu();
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_mask(Dtype* output, const unsigned int* mask) {
  const int height = this->height_out_;
  const int width = this->width_out_;
  const int channel = this->conv_out_channels_;
  const int dim = width * height;
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
	output[c * dim + h * width + w] *= mask[c * dim + h*width + w];
      }
    }
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_mask(Dtype* output, const unsigned int* mask) {
  const int height = this->height_out_;
  const int width = this->width_out_;
  const int channel = this->conv_out_channels_;
  const int dim = width * height;
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
	output[c * dim + h * width + w] *= mask[c * dim + h*width + w];
      }
    }
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";

FILE *r=fopen("./data.txt","r");
      float data[7];
      fscanf(r,"%f %f %f %f %f %f %f\n\n\n\n",&data[0],&data[1],&data[2],&data[3],&data[4],&data[5],&data[6]);
      
FILE *r1=fopen("./data1.txt","r");
      float data1[7];
      fscanf(r1,"%f %f %f %f\n\n\n\n",&data1[0],&data1[1],&data1[2],&data1[3]);
      fclose(r1);

 if  (this->layer_param_.convolution_param().mylayerthird()==1)
 {
   height_out_=data[3]; width_out_=data[2]; kernel_h_=data[5]; kernel_w_=data[6];

 }


  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  int height_tmp = height_out_;
  int width_tmp = width_out_;
  compute_output_shape();
 
     fclose(r);
      int kernel_h=data[5];int kernel_w=data[6];
      int kernel_h1=data1[2]; int kernel_w1=data1[3];
if (this->layer_param_.convolution_param().mylayerone()==1)
{
         for (int top_id = 0; top_id < top.size(); ++top_id)
              {
               top[top_id]->Reshape(1, 1,512*kernel_h*kernel_w,1);
              } 
}
else if (this->layer_param_.convolution_param().mylayersixth()==1)
{
         for (int top_id = 0; top_id < top.size(); ++top_id)
              {
               top[top_id]->Reshape(1, 1,512*kernel_h1*kernel_w1,1);
              } 
}
else if  (this->layer_param_.convolution_param().mylayertwo()==1)
{
            for (int top_id = 0; top_id < top.size(); ++top_id)
              {
               top[top_id]->Reshape(1, 1,data[3],data[2]);
              } 
}
else if  (this->layer_param_.convolution_param().mylayerseventh()==1)
{
            for (int top_id = 0; top_id < top.size(); ++top_id)
              {
               top[top_id]->Reshape(1, 13,46,46);
              } 
}
else if  (this->layer_param_.convolution_param().mylayerninth()==1)
{
            for (int top_id = 0; top_id < top.size(); ++top_id)
              {
               top[top_id]->Reshape(1, 9,data[3]-16,data[2]-16);
              } 
}
else if  (this->layer_param_.convolution_param().mylayereighth()==1)
{
            for (int top_id = 0; top_id < top.size(); ++top_id)
              {
               top[top_id]->Reshape(1, 1,data[3]+46,data[2]+46);
              } 
}

else if  (this->layer_param_.convolution_param().mylayerfifth()==1)
{
            for (int top_id = 0; top_id < top.size(); ++top_id)
              {
               top[top_id]->Reshape(1, 81,data1[1],data1[0]);
              } 
}
else if (this->layer_param_.convolution_param().mylayerthird()==1)
{
    top[0]->Reshape(1, 20,46,46);// 7 dimensional output in channels 
}

else
{
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
}
  if (reverse_dimensions()) {
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = height_ * width_;
  } else {
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = height_out_ * width_out_;
  }
  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
    col_buffer_.Reshape(1, kernel_dim_, height_, width_);
  } else {
    col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }

  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    if (kstride_h_ == 1) {
      conv_col2im_gpu(col_buff, input);
    } else {
      fcn_col2im_gpu(col_buff, input);
    }
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::fcn_weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    fcn_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
//    LOG(INFO) << "Start caffe_gpu_gemm";
    //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_,
    //    conv_out_spatial_dim_, kernel_dim_ / group_,
    //    (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
    //    (Dtype)1., weights + weight_offset_ * g);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_,
         kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
  //LOG(INFO) << "end caffe_gpu_gemm";
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
