clear;
clc;
ori_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/OriData_test/yuv_format/';
rec_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/rec_r04_test/yuv_format/';
pred_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/preds/r04_building_y/eval/';
new_ori_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/OriData_test/yuv_format/sameorder/';
% new_pred_path='/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/pointcloud/code/Enhancement_MPEG/data/preds/r06_manual/r06_manual_same/';
sequences=dir([pred_path,'*.ply']);
sequence_number=length(sequences);
for i=1:sequence_number
    ori_name=sequences(i).name;
    ori=pcread([ori_path,ori_name]);
    ori_loc=ori.Location;
    ori_col=ori.Color;
    ori_col_y=ori_col(:,1);
    rec=pcread([rec_path,ori_name(1:end-4),'_rec.ply']);
    rec_loc=rec.Location;
    rec_col=rec.Color;
    rec_col_y=rec_col(:,1);
    pred=pcread([pred_path,ori_name]);
    pred_loc=pred.Location;
    pred_col=pred.Color;
    pred_col_y=pred_col(:,1);
    kdtreeObj_ori=KDTreeSearcher(ori_loc,'distance','euclidean');
    [idx_rec,dis_rec]=knnsearch(kdtreeObj_ori,rec_loc,'k',1);
    new_ori_loc=ori_loc(idx_rec,:);
    new_ori_col=ori_col(idx_rec,:);
    new_ori_col_y=new_ori_col(:,1);
    pt_ori=pointCloud(new_ori_loc);
    pt_ori.Color=new_ori_col;
    pcwrite(pt_ori,[new_ori_path,ori_name],'PLYFormat','ascii');
    %% compute psnr for rec point cloud
    rec_error = double(new_ori_col_y)-double(rec_col_y);
    rec_residual=power( rec_error, 2 );
    rec_mse=mean(mean(rec_residual));
    rec_psnr=10*log10(255*255/rec_mse);
    fprintf('sequence: %s \n',ori_name);
    fprintf('ori_mse:%f \n',rec_mse);
    fprintf('ori_psnr:%f \n',rec_psnr);
    
    
%     [idx_pred,dis_pred]=knnsearch(kdtreeObj_ori,pred_loc,'k',1);
%     new_pred_loc=pred_loc(idx_pred,:);
%     new_pred_col=pred_col(idx_pred,:);
%     pt_pred=pointCloud(new_pred_loc);
%     pt_pred.Color=new_pred_col;
%     pcwrite(pt_pred,[new_pred_path,ori_name],'PLYFormat','ascii');
    %% compute psnr for pred point cloud
    pred_error = double(new_ori_col_y)-double(pred_col_y);
    pred_residual=power( pred_error, 2 );
    pred_mse=mean(mean(pred_residual));
    pred_psnr=10*log10(255*255/pred_mse);
    % printf('sequence: %s \n',ori_name);
    fprintf('pred_mse:%f \n',pred_mse);
    fprintf('pred_psnr:%f \n',pred_psnr);
end
    