clear;
clc,close all;
new_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/rec/rec_raht_r06_test/yuvFormat/';
%txt_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/rec/rec_r02_test/';
%h5_train='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/TrainData/r04_manual_yuv/';
% h5_test='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/TrainData/r06_manual/test/';
%txt_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/TrainData/r04_manual_yuv/TrainFile_Building.txt';
ori_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/rec/rec_raht_r06_test/';
% file=importdata(txt_path);
% new_path=[ori_path,'yuv_format/'];
sequences=dir([ori_path,'*.ply']);
sequence_number=length(sequences);
h1=waitbar(0,'read from sequences...');
for i=1:sequence_number 
    str1=['reading sequences...',num2str(i/sequence_number),'%'];
    waitbar( i/sequence_number,h1,str1);
    rec_name=sequences(i).name;
    % rec_name=file{i};
    
    rec_onlyName=rec_name(1:end-8);
    fprintf('The %d -th sequence：%s \n',i,rec_onlyName);
    % rec_Name=[rec_onlyName,'.ply'];
    rec=pcread([ori_path,rec_name]);
    rec_loc=rec.Location;
    rec_color=rec.Color;
    rec_color_yuv=rgb2yuv(rec_color);
    pt_new=pointCloud(rec_loc);
    pt_new.Color=rec_color_yuv;
    pcwrite(pt_new,[new_path,rec_name],'PLYFormat','ascii');
    
end
close(h1);