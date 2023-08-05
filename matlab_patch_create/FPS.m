function centroids=FPS(xyz,npoint)
% npoint=1024;
% pt=pcread('/home/vim/weiwei/pointcloud/dataset/longdress_vox10_1300.ply');
% xyz=pt.Location;
[N,C]=size(xyz);
centroids=zeros(npoint,1);
distance=ones(N,1)*1e10;
farthest=round(rand(1,1)*N);
for i=1:npoint
    centroids(i)=farthest;
    centroid=xyz(farthest,:);
    dist=sum((xyz-centroid).^2, 2);    % [N,1]
    mask=dist<distance;
    distance(mask)=dist(mask);
    [~,farthest]=max(distance);
end
end
% resampled_xyz=xyz(centroids,:);
% pt_new=pointCloud(resampled_xyz);
% color=pt.Color;
% pt_new.Color=color(centroids,:);
% pcwrite(pt_new,'/home/vim/weiwei/pointcloud/dataset/resample_londress.ply','PLYFormat','ascii');
% pcshow(pt_new)

