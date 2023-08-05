clc;
clear;
src_pth='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/DIV2K_Dataset/DIV2K_train_HR_sub/';
dst_pth='';
noise_pth='';
train_pth='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/TrainData/DIV2K_std0.02_y/';  % path to save h5 train data
imgs=dir([src_pth,'*.png']);
img_num=length(imgs);
h5create([train_pth,'DIV2K.h5'],'/data',[32,32,1,img_num]);
h5create([train_pth,'DIV2K.h5'],'/label',[32,32,1,img_num]);
box_label=zeros(32,32,1,img_num);
box_data=zeros(32,32,1,img_num);
for i=1:img_num
    ori_name=imgs(i).name;
    ori_only_name=ori_name(1:end-4);
    img=imread([src_pth,ori_name]);
    img_yuv=rgb2ycbcr(img);
    img_y=img_yuv(:,:,1);
    img_y_norm=double(img_y)/255.0;
    img_noise_norm=img_y_norm+0.02*randn(size(img_y_norm));    % mean:0, standard deviation:0.02
    box_data(:,:,:,i)=img_noise_norm;
    box_label(:,:,:,i)=img_y_norm;    
end
h5write([train_pth,'DIV2K.h5'],'/data',box_data);
h5write([train_pth,'DIV2K.h5'],'/label',box_label);
