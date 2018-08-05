function tmp=extract_features2(features,roi1)
% net=features{1}.fparams.net;
% roi_cnn=imresize(roi1,features{1}.img_sample_sz);
% % roi1 = bsxfun(@minus, roi1,imresize(net.meta.normalization.averageImage,[361 361]));
% roi_cnn = bsxfun(@minus, roi_cnn,net.meta.normalization.averageImage);
% cnn_feat = vl_simplenn(net,roi_cnn);
% features_output{1}=cnn_feat(1).x(features{1}.fparams.start_ind(1,1):features{1}.fparams.end_ind(1,1),features{1}.fparams.start_ind(1,1):features{1}.fparams.end_ind(1,1),:);
% features_output{2}=cnn_feat(4).x(features{1}.fparams.start_ind(2,1):features{1}.fparams.end_ind(2,1),features{1}.fparams.start_ind(2,1):features{1}.fparams.end_ind(2,1),:);
% features_output{1}=average_feature_region(features_output{1}, features{1}.fparams.downsample_factor(1));
% features_output{2}=average_feature_region(features_output{2}, features{1}.fparams.downsample_factor(2));


roi_hog=imresize(roi1,features{2}.img_sample_sz);
fparam.cell_size=4; fparam.useForColor=1; fparam.useForGray=1;
fparam.nDim=31; fparam.penalty=0; gparam.cell_size=4;
[ feature_image ] = get_fhog(roi_hog, fparam, gparam );
features_output{3}=feature_image;
fparam.tablename='CNnorm';fparam.useForGray=0;
fparam.cell_size=4; fparam.useForColor=1;fparam.nDim=10;
fparam.penalty=0;
gparam.normalize_power=2; gparam.normalize_size=1;
gparam.normalize_dim=1;gparam.square_root_normalization=0;
gparam.cell_size=4;
features_output{4}=get_table_feature( uint8(roi_hog), fparam, gparam);

  feature_map = cellfun(@(x) bsxfun(@times, x, ...
        ((size(x,1)*size(x,2))^gparam.normalize_size * size(x,3)^gparam.normalize_dim ./ ...
        (sum(abs(reshape(x, [], 1, 1, size(x,4))).^gparam.normalize_power, 1) + eps)).^(1/gparam.normalize_power)), ...
        features_output, 'uniformoutput', false);


features=cat(3, feature_map{3},  feature_map{4});
% features=cat(3,  feature_map{1},    feature_map{2});
% tmp=zeros(46,46,512);
tmp(:,:,1:41)=features;
tmp=single(tmp);
% tmp=permute(tmp,[2 1 3 4]);
a=1;


