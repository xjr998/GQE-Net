ori_path='../pointcloud/code/ARCNN-skipConnection/ARCNN-pytorch-master/data/oriDataset/';
noise_path='../pointcloud/code/ARCNN-skipConnection/ARCNN-pytorch-master/data/noise_data_0.03/';
sequences=dir([ori_path,'*.ply']);
sequence_number=length(sequences);
for i=1:sequence_number
    ori_name=sequences(i).name;
    ori_onlyName=ori_name(1:end-4);
    rec_name=[ori_onlyName,'_rec.ply'];
    ori=pcread([ori_path,ori_name]);
    ori_loc=ori.Location;
    ori_color=ori.Color;
    pointNumber=length(ori_loc);
    noise=0+0.03*randn(pointNumber,3);   % 产生均值为0，标准差为0.03的噪声
    color_temp=double(ori_color)/255+noise;
    rec_color=uint8(color_temp*255);
    pt_rec=pointCloud(ori_loc);
    pt_rec.Color=rec_color;
    pcwrite(pt_rec,[noise_path,rec_name],'PLYFormat','ascii');
end