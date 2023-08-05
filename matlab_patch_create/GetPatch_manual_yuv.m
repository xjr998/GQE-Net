clear;
clc,close all;
% ori_path='data_ori/';
% rec_base_path='data_r06/';
% h5_train='h5/';
% txt_path='data_ori/trainFile.txt';
ori_path='../lift/same_order/';
rec_base_path='../lift_rec/';
h5_train='h5/';
txt_path='../lift/testFile1.txt';
rates = {'_r01','_r02', '_r03', '_r04', '_r05', '_r06'};
% rates = {'_r01','_r02', '_r03', '_r04'};
rates = { '_r04', '_r05', '_r06'};

file=importdata(txt_path);
sequence_number=length(file);
mkdir(h5_train);
for i=1:sequence_number
    for k = 1:length(rates)
        ori_name=file{i};
        fprintf('The %d -th sequence- %s \n',i,ori_name);
        ori_onlyName=ori_name(1:end-4);
        write_name=[h5_train,ori_onlyName,rates{k},'.h5'];
        % tempSplit=strsplit(ori_onlyName,'_');
        rec_onlyName=[ori_onlyName,rates{k}];
        % rec_onlyName=[ori_onlyName,'_rec-',tempSplit{end}];
        ori=pcread([ori_path,ori_name]);
        ori_loc=ori.Location;
        ori_color=ori.Color;
        ori_color_yuv=rgb2yuv(ori_color);
        rec=pcread([rec_base_path,rec_onlyName,'.ply']);
        rec_color=rec.Color;
        rec_color_yuv=rgb2yuv(rec_color);
        rec_loc=rec.Location;
        
        pointNumber=length(rec_loc);
        num_Sample=round(pointNumber * 2 / 2048);
        
       
        h5create(write_name,'/data',[num_Sample,2048,6]);
        h5create(write_name,'/label',[num_Sample,2048,3]);
        box_label_train=zeros(num_Sample,2048,3);
        box_data_train=zeros(num_Sample,2048,6);
        
        centroids_rec=FPS(rec_loc,num_Sample);                        %FPS algorithm select the index of the represent point
        
        kdtreeObj_ori=KDTreeSearcher(ori_loc,'distance','euclidean');             % build the object for kdtree
        kdtreeObj_rec=KDTreeSearcher(rec_loc,'distance','euclidean');
        centroid_loc=rec_loc(centroids_rec,:);
        [idxnn_rec,dis]=knnsearch(kdtreeObj_rec,centroid_loc,'k',2048);   % 索引按照距离centroid的由小到大的顺序排列,[num_sample,2048]
        
        for j=1:num_Sample
            
            curPatchIdx_rec=idxnn_rec(j,:);        % 1024
            curPatchLoc_rec=rec_loc(curPatchIdx_rec,1:3);
            curPatchCol_rec=rec_color_yuv(curPatchIdx_rec,:);
            box_data_train(j,:,:) = [curPatchLoc_rec,curPatchCol_rec];
            [idx_ori,dis_ori]=knnsearch(kdtreeObj_ori,curPatchLoc_rec,'k',1);    % corresponding idx in reconstruction point cloud in the same patch with the original pt
            curPatchLoc_ori=ori_loc(idx_ori,:);
            curPatchCol_ori=ori_color_yuv(idx_ori,:);
            box_label_train(j,:,:)=curPatchCol_ori;
%             if(curPatchLoc_ori~=curPatchLoc_rec)
%                 error('rec not equal to ori when doing train dataset');
%             end
            
            
            
        end
        %     close(h2);
        clear f;
        h5write(write_name,'/data',box_data_train);
        h5write(write_name,'/label',box_label_train);
        
    end
end
% close(h1);
