function features=crop_feature(img,pos,fsolver,targetLoc)

num_pos=size(pos,1);

sbin=8;
if targetLoc(3)<30||targetLoc(4)<30&&max(targetLoc(3),targetLoc(4))<50
height=floor(targetLoc(4)/12*30); width=floor(targetLoc(3)/12*30);
else
height=107; width=107;
end
sz=[height width]; 
p=[pos(:,1)+pos(:,3)/2, pos(:,2)+pos(:,4)/2, pos(:,3), pos(:,4), zeros(num_pos,1)];
affinetotal=[p(:,1), p(:,2), p(:,3)/sz(1), p(:,5), p(:,4)./p(:,3), zeros(num_pos,1)];
affinetotal=double(affinetotal');


patch1=warpimg(double(img(:,:,1)),affparam2mat(affinetotal),sz);
patch2=warpimg(double(img(:,:,2)),affparam2mat(affinetotal),sz);
patch3=warpimg(double(img(:,:,3)),affparam2mat(affinetotal),sz);
feature_input = fsolver.net.blobs('data');
feature_blob4 = fsolver.net.blobs('conv4_3');

fsolver.net.set_input_dim([0, 1, 3, height,width ]);
for i=1:size(pos,1)
%     i
    patch=cat(3,patch1(:,:,i),patch2(:,:,i));
    patch=cat(3,patch,patch3(:,:,i));
    patch= impreprocess(patch);
    feature_input.set_data(single(patch));
    fsolver.net.forward_prefilled();
   tmp = feature_blob4.get_data();
   tmp=imresize(tmp,[3 3]);

   if i==1
       features=zeros(num_pos,length(tmp(:)));
   end
   features(i,:)=tmp(:);
end
fsolver.net.set_input_dim([0, 1, 3, 361,361 ]);