classdef Blob < handle
  % Wrapper class of caffe::Blob in matlab
  
  properties (Access = private)
    hBlob_self
  end
  
  methods
    function self = Blob(hBlob_blob)
      CHECK(is_valid_handle(hBlob_blob), 'invalid Blob handle');
      
      % setup self handle
      self.hBlob_self = hBlob_blob;
    end
    function shape = shape(self)
      shape = caffe_('blob_get_shape', self.hBlob_self);
    end
    function reshape(self, shape)
      shape = self.check_and_preprocess_shape(shape);
      caffe_('blob_reshape', self.hBlob_self, shape);
    end
    function data = get_data(self)
      data = caffe_('blob_get_data', self.hBlob_self);
    end
    function set_data(self, data)
      data = self.check_and_preprocess_data(data);
      caffe_('blob_set_data', self.hBlob_self, data);
    end
      function set_data1(self, channel,height,width,height_in,width_in,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,data,cos_win)
%       data = self.check_and_preprocess_data(data);
      caffe_('my_set_data', self.hBlob_self, single(channel),single(height),single(width),single(height_in),single(width_in),single(kernel_h),single(kernel_w),single(pad_h),single(pad_w),single(stride_h),single(stride_w),single(data),single(cos_win));
      end
      
       function set_data8(self, tmp_large)
%       data = self.check_and_preprocess_data(data);
      caffe_('my_set_data8', self.hBlob_self, single(tmp_large));
       end
       
          function set_raw_feature(self,data1,data2)
%       data = self.check_and_preprocess_data(data);
      caffe_('set_raw_feature', self.hBlob_self, single(data1),single(data2));
       end
      
        function [data1 data2]=get_pad_feature(self)
%       data = self.check_and_preprocess_data(data);
      [data1 data2]=caffe_('get_pad_feature', self.hBlob_self);
       end
       
        function set_momentum(self, momentum)
%       data = self.check_and_preprocess_data(data);
      caffe_('set_momentum', self.hBlob_self, single(momentum));
      end
       
      function set_data9(self, tmp_refine)
%       data = self.check_and_preprocess_data(data);
      caffe_('my_set_data9', self.hBlob_self, single(tmp_refine));
      end
      
      function set_data2(self, feature,lambda1,lambda2,weight1,weight2)
%       data = self.check_and_preprocess_data(data);
      caffe_('my_set_data2', self.hBlob_self, single(feature),single(lambda1),single(lambda2),single(weight1),single(weight2));
      end
     function set_data5(self, channel1,kernel_h1,kernel_w1,height1,width1,pad_h1,pad_w1,stride_h1,stride_w1,tmp1)
%       data = self.check_and_preprocess_data(data);
      caffe_('my_set_data5', self.hBlob_self, single(channel1),single(kernel_h1),single(kernel_w1),single(height1),single(width1),...
          single(pad_h1),single(pad_w1),single(stride_h1),single(stride_w1),single(tmp1));
     end
     
      function set_data7(self, x,x1,y,y1)
%       data = self.check_and_preprocess_data(data);
      caffe_('my_set_data7', self.hBlob_self, single(x),single(x1),single(y),single(y1));
      end
     
      function [data1,data2,data3,data4,data5,data6,data7] = my_get_data(self)
      [data1,data2,data3,data4,data5,data6,data7] = caffe_('my_get_data', self.hBlob_self);
    end
      
    function diff = get_diff(self)
      diff = caffe_('blob_get_diff', self.hBlob_self);
    end
    function set_diff(self, diff)
      diff = self.check_and_preprocess_data(diff);
      caffe_('blob_set_diff', self.hBlob_self, diff);
    end
  end
  
  methods (Access = private)
    function shape = check_and_preprocess_shape(~, shape)
      CHECK(isempty(shape) || (isnumeric(shape) && isrow(shape)), ...
        'shape must be a integer row vector');
      shape = double(shape);
    end
    function data = check_and_preprocess_data(self, data)
      CHECK(isnumeric(data), 'data or diff must be numeric types');
      self.check_data_size_matches(data);
      if ~isa(data, 'single')
        data = single(data);
      end
    end
    function check_data_size_matches(self, data)
      % check whether size of data matches shape of this blob
      % note: matlab arrays always have at least 2 dimensions. To compare
      % shape between size of data and shape of this blob, extend shape of
      % this blob to have at least 2 dimensions
      self_shape_extended = self.shape;
      if isempty(self_shape_extended)
        % target blob is a scalar (0 dim)
        self_shape_extended = [1, 1];
      elseif isscalar(self_shape_extended)
        % target blob is a vector (1 dim)
        self_shape_extended = [self_shape_extended, 1];
      end
      % Also, matlab cannot have tailing dimension 1 for ndim > 2, so you
      % cannot create 20 x 10 x 1 x 1 array in matlab as it becomes 20 x 10
      % Extend matlab arrays to have tailing dimension 1 during shape match
      data_size_extended = ...
        [size(data), ones(1, length(self_shape_extended) - ndims(data))];
      is_matched = ...
        (length(self_shape_extended) == length(data_size_extended)) ...
        && all(self_shape_extended == data_size_extended);
      CHECK(is_matched, ...
        sprintf('%s, input data/diff size: [ %s] vs target blob shape: [ %s]', ...
        'input data/diff size does not match target blob shape', ...
        sprintf('%d ', data_size_extended), sprintf('%d ', self_shape_extended)));
    end
  end
end
