clear;
clc;
ori_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/TMC/MPEG_CTC/Cat1_A/';
txt_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/TMC/MPEG_CTC/Cat1_A/trainFile.txt';
h5_train='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/TrainData/std0.05_y/';
%rec_path='/home/vim/weiwei/pointcloud/Enhancement_MPEG/data/TrainData/r04_manual_yuv/r04_manual_train_people/yuv_format/';
% pred_path='/home/vim/weiwei/pointcloud/Enhancement_MPEG/data/preds/r04_people_y/eval/';
% new_ori_path=[ori_path, 'sameorder/'];
% new_pred_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/preds/r06_manual/r06_manual_same/';
files=importdata(txt_path);
% sequences=dir([ori_path,'*.ply']);
sequence_number=length(files);
for i=1:sequence_number
    % ori_name=sequences(i).name;
    ori_name=files{i};
    ori_only_name=ori_name(1:end-4);
    ori=pcread([ori_path,ori_name]);
    ori_loc=ori.Location;
    centroid=FPS(ori_loc, 1024);
    ori_col=ori.Color;
    pointNumber=length(ori_loc);
    num_Sample=round(pointNumber/1024);
    centroids_ori=FPS(ori_loc,num_Sample); 
    centroids_loc=ori_loc(centroids_ori,:);
    kdtreeObj_ori=KDTreeSearcher(ori_loc,'distance','euclidean');
    [idx,dis]=knnsearch(kdtreeObj_ori,centroids_loc,'k',1024);
    ori_color_yuv=rgb2yuv(ori_col);
    ori_col_normal=double(ori_col)/255;
    noise_color_normal=double(ori_col_normal)+0.05*randn(size(ori_col));   % [0,0.01]normal distribution
    noise_color=noise_color_normal*255;
    noise_color_yuv=rgb2yuv(noise_color);
    h5create([h5_train,ori_only_name,'.h5'],'/data',[1024,1,num_Sample]);
    h5create([h5_train,ori_only_name,'.h5'],'/label',[1024,1,num_Sample]);
    box_label=zeros(1024,1,num_Sample);
    box_data=zeros(1024,1,num_Sample);
    for j=1:num_Sample
        CurIdx=idx(j,:);
        CurColor_label=ori_color_yuv(CurIdx,1);
        CurColor_data=noise_color_yuv(CurIdx,1);
        box_label(:,:,j)=CurColor_label;
        box_data(:,:,j)=CurColor_data;
    end
    h5write([h5_train,ori_only_name,'.h5'],'/data',box_data);
    h5write([h5_train,ori_only_name,'.h5'],'/label',box_label);
end