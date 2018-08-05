function KRR_VOT
cleanupObj = onCleanup(@cleanupFun);
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));
[handle, image1, init_rect] = vot('rectangle');
im1 = double(imread(image1));
[init_rect] = region2location(im1, init_rect);
set_tracker_param;
num_z = 4;
im1 = double(imread(image1));

if size(im1,3)~=3
    im1(:,:,2) = im1(:,:,1);
    im1(:,:,3) = im1(:,:,1);
end
[roi1,~,~,~,roi2,roi_large,pad_large,roi_pos_large] = ext_roi(im1, location, center_off,  roi_size, roi_scale_factor);

xl=extract_features2(features,roi1);
xl=permute(xl,[2 1 3 4]);
cos_win_large = single(hann(52) * hann(52)');
cos_win_large=sqrt(cos_win_large);
cos_win_large=sqrt(cos_win_large);
xl=bsxfun(@times,xl,cos_win_large);
xl=imresize(xl,[46 46],'nearest');

tmp_height=(move_height-1)+kernel_h/3;
tmp_width=(move_width-1)+kernel_w/3;
cnna_input = cnna.net.blobs('data');
fea_sz=[46 46];
cos_win = single(hann(fea_sz(1)) * hann(fea_sz(2))');
cos_win=sqrt(cos_win);
cos_win=sqrt(cos_win);
cos_win=permute(cos_win,[2 1 3]);
cnna_input.set_data1(512,46,46,kernel_h,kernel_w,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,zeros(tmp_width,tmp_height,512,9),cos_win);
%% extract vgg feature
roi1 = impreprocess(roi1);

width_in=floor(46/ roi_scale_factor(1)); height_in=floor( 46/ roi_scale_factor(2) );

fsolver.net.set_net_phase('test');
feature_input = fsolver.net.blobs('data');
feature_blob4 = fsolver.net.blobs('conv4_3');

fsolver.net.set_input_dim([0, 1, 3, roi_size, roi_size]);
feature_input.set_data(single(roi1));
fsolver.net.forward_prefilled();
deep_feature1 = feature_blob4.get_data();

fea_sz = size(deep_feature1);
cos_win = single(hann(fea_sz(1)) * hann(fea_sz(2))');
cos_win=sqrt(cos_win);
cos_win=sqrt(cos_win);
deep_feature1 = bsxfun(@times, deep_feature1, cos_win);

%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% initialization
cnna.net.set_net_phase('train');
spn.net.set_net_phase('train');
cnna.net.set_input_dim([0, 1, fea_sz(3), fea_sz(2), fea_sz(1)]);
%% prepare training samples
map1 =  GetMap(size(im1), fea_sz, roi_size, location, center_off, roi_scale_factor, map_sigma_factor, 'trans_gaussian');
map1 = permute(map1, [2,1,3]);
map1 = repmat(single(map1), [1,1,ensemble_num]);
blob(:,:,:,1)=deep_feature1(:,:,:);
blob(:,:,:,2)=deep_feature1(:,:,:);
kernel_h=max(3*floor(height_in/3),3); kernel_w=max(3*floor(width_in/3),3); stride_w=1; stride_h=1; pad_h=floor(kernel_h/2); pad_w=floor(kernel_w/2);

if seq.init_rect(3)>30&&seq.init_rect(4)>60
   kernel_w=max(3*floor(width_in/3)-3,3); 
     kernel_h=max(3*floor(height_in/3)-3,3);
else
    kernel_w=max(3*ceil(width_in/3)-3,3); 
     kernel_h=max(3*ceil(height_in/3)-3,3);
end

cnn_my_input = cnn_my.net.blobs('data');
[height, width]=size(deep_feature1(:,:,1));

basis=permute(xl,[2 1 3]);

basis= padarray(basis,[ pad_h,pad_w],0);

[deep_h1,deep_w1]=size(basis(:,:,1));

   for i=1:3
       for j=1:3
          layer_index=3*(i-1)+j;
           tmp(:,:, :,layer_index)=basis(  (i-1)*kernel_h/3+1:deep_h1-(3-i)*kernel_h/3,  (j-1)*kernel_w/3+1:deep_w1-(3-j)*kernel_w/3,: );
       end
   end
tmp=permute(tmp,[2 1 3 4]);
cnna_input.set_raw_feature(xl/200,tmp/100);

lambda1=1; lambda2=0;

weight1=zeros(1,move_width*move_height); weight2=zeros(81,1);
cnna_input.set_data2(deep_feature1,lambda1,lambda2,weight1,weight2);
cnn_my_input.set_data2(deep_feature1,lambda1,lambda2,weight1,weight2);

blob(:,:,:,1)=deep_feature1(:,:,:);
blob(:,:,:,2)=deep_feature1(:,:,:);

number_h=floor((height+2*pad_h-kernel_h)/stride_h)+1;
number_w=floor((width+2*pad_w-kernel_w)/stride_w)+1;

map2 =  GetMap(size(im1), [number_h number_w 1], roi_size, location, center_off, roi_scale_factor, map_sigma_factor, 'trans_gaussian');
map2 = permute(map2, [2,1,3]);

map3 =  GetMap(size(im1), [46 46 1], roi_size, location, center_off, roi_scale_factor, map_sigma_factor, 'trans_gaussian');
map3 = permute(map3, [2,1,3]);
map3 =repmat(map3,[1 1 52]);

map4 =  GetMap(size(im1), [46 46 1], roi_size, location, center_off, roi_scale_factor, map_sigma_factor, 'trans_gaussian');
map4 = permute(map4, [2,1,3]);
map4 =repmat(map4,[1 1 52]);

for i=1:13
    rand1=rand(1)-0.5;
    sign=1;
    if rand1~=0
      sign=rand1/abs(rand1);
    end
        transform_x=0;
         transform_y=0;

     
     map3(:,:,4*i-3:4*i)=circshift(map3(:,:,4*i-3:4*i),[transform_x, transform_y]);
             for index=4*i-3:4*i
                transform_x_total{index}=transform_x;
                transform_y_total{index}=transform_y;
              end
end
basis=permute(deep_feature1,[2 1 3]);

basis= padarray(basis,[ pad_h1,pad_w1],0);

[deep_h1 deep_w1]=size(basis(:,:,1));

[x,x1,y,y1]=generate_displacement();
cnna_input.set_data7(x,x1,y,y1);
fail=0;
cnna_input.set_momentum(0);
deacy=1;
realiable=1;
lambda1=1; lambda2=0;
weight1=zeros(1,move_width*move_height); weight2=zeros(81,1);
cnna_input.set_data2(deep_feature1,lambda1,lambda2,weight1,weight2);
clear tmp;
fail=0;
cnna_input.set_momentum(0);
deacy=1;
realiable=1;


cnn_my_input=cnn_my.net.blobs('data');
cnn_my_output=cnn_my.net.blobs('conv5_f2');

for i=1:280
    spn.net.empty_net_param_diff();
    cnna.net.empty_net_param_diff();
    pre_heat_map1 = cnna.net.forward({deep_feature1/2000 deep_feature1});
     map_c=cnnc.net.forward({xl});
     diff_c=map_c{1}-map4;
     cnnc.net.backward({diff_c});
     cnnc.apply_update();
    %% regression 2
    map_regression2=pre_heat_map1{2};
    
     diff_regression2=map_regression2-map3;

    pre_heat_map = pre_heat_map1{1};
    
    diff_cnna = (pre_heat_map-map2);

    cnna.net.backward({single(diff_cnna*deacy) single(diff_regression2)});
    weight_total1{i}=cnna.net.params('conv5_f1', 1).get_data() ;% set weights
    weight_total2{i}=cnna.net.params('conv5_f1', 2).get_data() ; % set bias
    weight_total3{i}=cnna.net.params('conv5_f2', 1).get_data() ;% set weights
    weight_total4{i}=cnna.net.params('conv5_f2', 2).get_data() ; % set bias
    diff_cnna_total(i)=sum(abs(diff_cnna(:)));
     
    if abs(sum(diff_cnna(:)))>60
        realiable=0;
        deacy=deacy*0.5;
        fail=1;
        cnna.net.params('conv5_f1', 1).set_data(weight_total1{ max(i-8,1)}) ;% set weights
        cnna.net.params('conv5_f1', 2).set_data(weight_total2{max(i-8,1)}) ; % set bias
        cnna.net.params('conv5_f2', 1).set_data(weight_total3{max(i-8,1)}) ;% set weights
        cnna.net.params('conv5_f2', 2).set_data(weight_total4{max(i-8,1)}) ; % set bias
        cnna_input.set_momentum(1);
    end
    
    if fail==1
         cnna_input.set_momentum(1);
         cnna.apply_update();
         fail=0;
    else
         cnna.apply_update();
    end
   
    cnna_input.set_momentum(0);

    fprintf('Iteration %03d/%03d, CNN-A Loss %0.1f, SPN Loss %0.1f\n', i, max_iter, sum(abs(diff_cnna(:))), sum(abs(diff_regression2(:))));   
    last_loss = sum(abs(diff_cnna(:)));      
end
output=cnna.net.blobs('conv5_f1');
output1=output.get_data();

weight2=[0;-0.5;0;-0.5];
weight2=repmat(weight2,[1 9]);

pre_heat_map1=map_regression2;

for i=1:13
    map_tmp_tmp(:,:,i)=sum(pre_heat_map1(:,:,(i-1)*4+1:i*4),3);
end
for i=1:100
    map_dt=cnn_dtpooling.net.forward({map_tmp_tmp});
    map_dt=map_dt{1};
    diff_dt=map_dt-4*map3(:,:,1:13);
    cnn_dtpooling.net.backward({diff_dt});
    fprintf('Iteration %03d/%03d, CNN-A Loss %0.1f\n', i, max_iter, sum(abs(diff_dt(:))));   
     cnn_dtpooling.apply_update();
end

opts.batchSize_test = 256;
opts.bbreg_nSamples = 1000;
opts.imgSize=size(im1);
opts.scale_factor=1.02;
global kkk;
targetLoc=[seq.init_rect(1) seq.init_rect(2) seq.init_rect(3) seq.init_rect(4)  ];
try
   pos_examples = gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.3, 10);
   r = overlap_ratio(pos_examples,targetLoc);   
   pos_examples = pos_examples(r>0.6,:);
   pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
   features1=crop_feature(im1,pos_examples,fsolver1,targetLoc);
catch
    pos_examples = gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.3, 10);
    r = overlap_ratio(pos_examples,targetLoc);   
    pos_examples = pos_examples(r>0.6,:);
    pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);
    features1=crop_feature(im1,pos_examples,fsolver1,targetLoc);
end
X=features1;
bbox = pos_examples;
bbox_gt = repmat(targetLoc,size(pos_examples,1),1);
bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
close all
spn.net.set_input_dim([0, scale_param.number_of_scales_test, fea_sz(3), fea_sz(2), fea_sz(1)]);
start_frame = 1;
weight1=cnna.net.params('conv5_f1', 1).get_data() ;
weight2=cnna.net.params('conv5_f2', 1).get_data() ;
weight_inner1=cnna.net.params('conv5_f1', 1).get_data() ;
weight_inner2=cnna.net.params('conv5_f2', 1).get_data() ;
model_deep_feature=deep_feature1/2000;
model_raw_feature=xl/200;
model_deep_feature_slow=deep_feature1/2000;
model_weight1=weight1; model_weight2=weight2;
model_deep_feature_inner=deep_feature1/2000;
model_weight_inner1=weight_inner1; model_weight_inner2=weight_inner2;
close all;
w_spatial=zeros(1,400);
indicate_map=1;
im2_id=1;

while true
if im2_id==1
    im2_name= image1;
    im2_id=im2_id+1;
else
im2_id=im2_id+1;
 [handle, im2_name] = handle.frame(handle);
end
 if isempty(im2_name) % Are we done?
        break;
    end;
im2_id
   lambda1=0; lambda2=1;
   cnna_input.set_data2(model_deep_feature,lambda1,lambda2,model_weight1,model_weight2);
   cnna.net.set_net_phase('test');
   spn.net.set_net_phase('test');
   fprintf('Processing Img: %d\t', im2_id);
    im2 = double(imread(im2_name));
    if size(im2,3)~=3
        im2(:,:,2) = im2(:,:,1);
        im2(:,:,3) = im2(:,:,1);
    end 
    [roi2, roi_pos, padded_zero_map, pad,roi_2] = ext_roi(im2, location, center_off,  roi_size, roi_scale_factor);
    %% preprocess roi
    xl=extract_features2(features,roi2);
    xl=permute(xl,[2 1 3 4]);
    xl=bsxfun(@times,xl,cos_win_large);
    xl=imresize(xl,[46 46]);
    roi2 = impreprocess(roi2);
    feature_input.set_data(single(roi2));
    fsolver.net.forward_prefilled();
    deep_feature2 = feature_blob4.get_data();
    deep_feature2 = bsxfun(@times, deep_feature2, cos_win);
    basis=permute(xl,[2 1 3]);
    basis= padarray(basis,[ pad_h,pad_w],0);            
            for i=1:3
                for j=1:3
                    layer_index=3*(i-1)+j;
                    tmp(:,:, :,layer_index)=basis(  (i-1)*kernel_h/3+1:deep_h1-(3-i)*kernel_h/3,  (j-1)*kernel_w/3+1:deep_w1-(3-j)*kernel_w/3,: );                    
                end
            end
            cnna_input.set_raw_feature(model_raw_feature,tmp/100);
            clear tmp;

    %% compute confidence map
    cnna_input.set_data2(model_deep_feature,lambda1,lambda2,model_weight1,model_weight2);
    pre_heat_map_total = cnna.net.forward({deep_feature2/2000 deep_feature2});
    cnna_input.set_data2(model_deep_feature_inner,lambda1,lambda2,model_weight_inner1,model_weight_inner2);
    cnn_my_input.set_data(deep_feature2/2000);
    cnn_my.net.forward_prefilled();
    pre_heat_map_my=cnn_my_output.get_data();
    pre_heat_map_my=pre_heat_map_my(:,:,1);
    pre_heat_map_my=permute(pre_heat_map_my,[2,1,3,4]);
    pre_heat_map = permute(pre_heat_map_total{1}, [2,1,3,4]);
    pre_heat_map=pre_heat_map(:,:,1);
    pre_heat_map= pre_heat_map(:,:,1);
    pre_heat_map1=pre_heat_map_total{2};
    map_c=cnnc.net.forward({xl});
    map_c=map_c{1};
     for i=1:13
             map_tmp_tmp1(:,:,i)=sum(map_c(:,:,(i-1)*4+1:i*4),3);
     end
     clear  map_tmp;
        for i=1:13
             map_tmp_tmp(:,:,i)=sum(pre_heat_map1(:,:,(i-1)*4+1:i*4),3);
        end

   for i=1:13
       gamma1=-w_spatial(4*i-3); gamma2=-w_spatial(4*i-2); gamma3=-w_spatial(4*i-1); gamma4=-w_spatial(4*i);
       if im2_id==1
        gamma1=0.1; gamma2=0; gamma3=0.1; gamma4=0;
       end
      gamma1=max(gamma1,0.001); gamma3=max(gamma3,0.001);
       if gamma1<0
           gamma1=0;
       end
       if gamma2<0
           gamma2=0;
       end 
       transform_x=transform_x_total{i}; transform_y=transform_y_total{i};
       try
           map_tmp(:,:,i)=circshift(map_tmp_tmp(:,:,i),[-transform_x -transform_y]);
       catch
           a=1;
       end

   end

  map=cnn_dtpooling.net.forward({map_tmp});
  map=map{1};
  map=permute(map,[2 1 3]);
  map_c=cnn_dtpooling.net.forward({map_tmp_tmp1});
  map_c=map_c{1};
  map_c=permute(map_c,[2 1 3]);
   if im2_id<11
       if realiable==0
        map_total=100*imresize(pre_heat_map,[46 46])+2*sum(map,3);
       else
           map_total=100*imresize(pre_heat_map,[46 46]);
       end
   else
        if realiable==1
           map_total=(70*imresize(pre_heat_map,[46 46])+30*imresize(pre_heat_map_my,[46 46]))*2+0.45*sum(map,3)+1*sum(map_c,3);
        else
           map_total=(70*imresize(pre_heat_map,[46 46])+30*imresize(pre_heat_map_my,[46 46]))*2+1*sum(map,3)+1*sum(map_c,3);    
        end
   end
search_large=0;
[height_im, width_im]=size(im1(:,:,1));
ratio1=height_im/240;  ratio2=width_im/320;
if sqrt(location(3).^2+location(4).^2)/ratio1/ratio2>40
    search_large=1;
        map_total=bsxfun(@times,map_total,cos_win.^(2));
else
        map_total=bsxfun(@times,map_total,cos_win.^(1.5));
end

    pre_heat_map1 = sum(pre_heat_map1(:,:,:), 3);
    %% compute local confidence
    pre_heat_map_upscale = imresize(map_total, roi_pos(4:-1:3))/100;
    pre_img_map = padded_zero_map;
    pre_img_map(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = pre_heat_map_upscale;
    pre_img_map = pre_img_map(pad+1:end-pad, pad+1:end-pad);
    [center_y, center_x] = find(pre_img_map == max(pre_img_map(:)));
    sorted_score=sort(pre_img_map(:),'descend');
  
  for candidate_id=1:5
       [tmp1,tmp2] = find(pre_img_map == sorted_score(candidate_id+1));
       center_y_can(candidate_id)=mean(tmp1); center_x_can(candidate_id)=mean(tmp2);
       location_can(candidate_id,:)=[center_x_can(candidate_id)-location(3)/2  center_y_can(candidate_id)-location(4)/2 location([3,4])];
  end
    center_x = mean(center_x);
    center_y = mean(center_y);
    if im2_id>2
    move = max(pre_heat_map(:)) >  0.01;
    else
        move=0;
    end
    max_confidence(im2_id)=max(pre_heat_map1(:));
       
    if move
        if move
            base_location = [center_x - location(3)/2, center_y - location(4)/2, location([3,4])];
        else
            base_location = location;
        end
      
          recovered_scale=5;
        
        scale_param.currentScaleFactor = scale_param.scaleFactors_test(recovered_scale);
        target_sz = location([3, 4]) * scale_param.currentScaleFactor;       
        location = [center_x - floor(target_sz(1)/2), center_y - floor(target_sz(2)/2), target_sz(1), target_sz(2)];
               for candidate_id=1:5
                     location_can(candidate_id,:)=[center_x_can(candidate_id)-floor(target_sz(1)/2)  center_y_can(candidate_id)-floor(target_sz(2)/2) target_sz];
               end
    else
         recovered_scale = (scale_param.number_of_scales_test+1)/2;
        scale_param.currentScaleFactor = scale_param.scaleFactors_test(recovered_scale);
    end
   
 bbox_=[location; location_can];
 X_=crop_feature(im2,bbox_,fsolver1,targetLoc);
 pred_boxes = predict_bbox_regressor(bbox_reg.model, X_, bbox_);

 ratio1_init=seq.init_rect(3)/seq.init_rect(4); ratio2_init=seq.init_rect(4)/seq.init_rect(3);
  if max(pre_heat_map1(:))>20&&(max(pre_heat_map(:))>0.06||~realiable)
     tmp=(location(:,1)-location(1,1)).^2+(location(:,2)-location(1,2)).^2<10;
  aa= double( round(mean(pred_boxes(tmp,:),1)));
  ratio1=aa(3)/aa(4); ratio2=aa(4)/aa(3);
  ratio1_init=seq.init_rect(3)/seq.init_rect(4); ratio2_init=seq.init_rect(4)/seq.init_rect(3);
  if (ratio1/ratio1_init)<1.2&&(ratio2/ratio2_init)<1.2&&im2_id>2
      location=aa;
  else
      a=1;
  end
  use_bbr=1;
  clear tmp;
  end
 if location(3)<width_im/11&&location(3)<seq.init_rect(3)&&max(ratio1_init,ratio2_init)<1.5
    center_x=location(1)+location(3)/2;
     location(3)=min(width_im/11,seq.init_rect(3));
     location(1)=center_x-location(3)/2;
 end 
 if location(4)<height_im/11&&location(4)<seq.init_rect(4)&&max(ratio1_init,ratio2_init)<1.5
     center_y=location(2)+location(4)/2;
     location(4)=min(height_im/11,seq.init_rect(4));
     location(2)=center_y-location(4)/2;
 end
  
    fprintf(' scale = %f\n', scale_param.scaleFactors_test(recovered_scale));

    randval=rand(1);
    if  im2_id < start_frame -1 + 30 && max(pre_heat_map(:))> 0.00 &&randval>0.0|| im2_id < start_frame -1 + 6
        update = true;
    elseif im2_id >= start_frame -1 + 30 && move && max(pre_heat_map(:))> 0.00&&randval>0.0
            update = true;
    else
        update = false;
    end
  
    update=1;
    if  update&&im2_id>2
          lambda1=1; lambda2=0;
        cnna_input.set_data2(model_deep_feature,lambda1,lambda2,model_weight1,model_weight2);
        
        [roi2 ,roi_pos,preim, pad,roi_2]= ext_roi(im2, location, center_off,  roi_size, roi_scale_factor);
        
        xl=extract_features2(features,roi2);
        xl=permute(xl,[2 1 3 4]);
        xl=bsxfun(@times,xl,cos_win_large);
        xl=imresize(xl,[46 46]);
        
        roi2 = impreprocess(roi2);
        feature_input.set_data(single(roi2));
        fsolver.net.forward_prefilled();
        deep_feature2 = feature_blob4.get_data();
        deep_feature2 = bsxfun(@times, deep_feature2, cos_win);

        cnna.net.set_net_phase('train');
           basis=permute(xl,[2 1 3]);
           basis= padarray(basis,[ pad_h,pad_w],0);
           [deep_h1 deep_w1]=size(basis(:,:,1));
            for i=1:3
                for j=1:3
                    layer_index=3*(i-1)+j;
                    tmp(:,:, :,layer_index)=basis(  (i-1)*kernel_h/3+1:deep_h1-(3-i)*kernel_h/3,  (j-1)*kernel_w/3+1:deep_w1-(3-j)*kernel_w/3,: );                    
                end
            end
            tmp=permute(tmp,[2 1 3,4]); 
            cnna_input.set_raw_feature(xl/200,tmp/100);
            clear tmp;
        for ii = 1:1
            cnna.net.empty_net_param_diff();
            pre_heat_map_train = cnna.net.forward({deep_feature2/2000 deep_feature2});
           
            pre_heat_map_train1 = pre_heat_map_train{1};

            diff_cnna = (pre_heat_map_train1-map2);
           
            pre_heat_map_train2 = pre_heat_map_train{2};
            diff_my = (pre_heat_map_train2-map3); 

            if (sum(abs(diff_cnna(:)))>100)
               continue;
            end
            cnna.net.backward({1*diff_cnna*deacy 1*diff_my});
           
            cnna.apply_update();
          
            fprintf('Iteration %03d/%03d, CNN-A Loss %0.1f, SPN Loss %0.1f\n', i, max_iter, sum(abs(diff_cnna(:))), sum(abs(diff_my(:))));  
        end
        for ii=1:3
              map_c=cnnc.net.forward({xl});
              diff_c=map_c{1}-map4;
              cnnc.net.backward({diff_c});
              cnnc.apply_update();
        end
        
        cnna.net.set_net_phase('test');
         weight1=cnna.net.params('conv5_f1', 1).get_data() ;
         weight2=cnna.net.params('conv5_f2', 1).get_data() ;
          model_deep_feature_slow=0.999*model_deep_feature_slow+0.001*deep_feature2/2000;
        if im2_id<11
            model_deep_feature=0.8*model_deep_feature+0.2*deep_feature2/2000;
            model_weight1=0.8*model_weight1+0.2*weight1; 
            model_weight2=0.8*model_weight2+0.2*weight2;
            model_deep_feature_inner=0.8*model_deep_feature_inner+0.2*deep_feature2/2000;
            model_weight_inner1=0.8*model_weight_inner1+0.2*weight_inner1; 
            model_weight_inner2=0.8*model_weight_inner2+0.2*weight_inner2;
            model_raw_feature=0.8*model_raw_feature+0.2*xl/200;
        elseif indicate_map==1
            model_weight1=0.999*model_weight1+0.001*weight1; 
            model_weight2=0.999*model_weight2+0.001*weight2;
            model_raw_feature=0.9*model_raw_feature+0.1*xl/200;
        if mod(im2_id-1,5)==1
            model_deep_feature_inner=0.9*model_deep_feature_inner+0.1*deep_feature2/2000;
            model_weight_inner1=0.999*model_weight_inner1+0.001*weight_inner1; 
            model_weight_inner2=0.999*model_weight_inner2+0.001*weight_inner2;
        end
            mask_update=cos_win>0.9;
             if mod(im2_id-1,5)==1&&max(pre_heat_map(:))>0.35
               tmp1=0.999*bsxfun(@times, model_deep_feature, 1-mask_update)+0.001*bsxfun(@times,deep_feature2/2000,1-mask_update);
               tmp2=0.9*bsxfun(@times, model_deep_feature, mask_update)+0.1*bsxfun(@times,deep_feature2/2000,mask_update);
               model_deep_feature=tmp2+tmp1;
             else
                model_deep_feature=0.999*model_deep_feature+0.001*deep_feature2/2000;  
            end
        
        else
            model_deep_feature=0.9*model_deep_feature+0.1*deep_feature2/2000;
            model_weight1=0.9*model_weight1+0.1*weight1; 
            model_weight2=0.9*model_weight2+0.1*weight2;
            model_deep_feature_inner=0.9*model_deep_feature_inner+0.1*deep_feature2/2000;
            model_weight_inner1=0.9*model_weight_inner1+0.1*weight_inner1; 
            model_weight_inner2=0.9*model_weight_inner2+0.1*weight_inner2;
        end
    end
%     figure(1)
%     Drwa resutls
   
%        if im2_id == start_frame+1,  %first frame, create GUI
%            figure('Name','Tracking Results');
% % %             figure(1)
%            im_handle = imshow(uint8(im2), 'Border','tight', 'InitialMag', 100 + 100 * (length(im2) < 500));
%             
%            rect_handle = rectangle('Position', location, 'EdgeColor','r', 'linewidth', 2);
%            text_handle = text(10, 10, sprintf('#%d',im2_id));
%            set(text_handle, 'color', [1 1 0], 'fontsize', 16, 'fontweight', 'bold');
% %              imwrite(frame2im(getframe(gcf)),sprintf('/media/wayne/h/vot-toolkit-master_2017/workspace/%04d.bmp',im2_id));
%        else
%            set(im_handle, 'CData', uint8(im2))
%            set(rect_handle, 'Position', location)
%            set(text_handle, 'string', sprintf('#%d',im2_id));
% %                imwrite(frame2im(getframe(gcf)),sprintf('/media/wayne/h/vot-toolkit-master_2017/workspace/%04d.bmp',im2_id));
%        end
%          drawnow
      if im2_id>2
          if(sum(isnan(location))>1)
              location=zeros(1,4);
          end
        handle = handle.report(handle, location);
      end
end
handle.quit(handle);
end









