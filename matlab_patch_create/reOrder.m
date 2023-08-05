clear;
clc;
% ori_path='../lift_building/';
% rec_path='../lift_rec_building/';
ori_path='../lift/dep12/';
rec_path='../lift_rec/';
reordered_recPath='../lift/same_order/';
sequences=dir([ori_path,'*.ply']);

sequence_number=length(sequences);
for i=1:sequence_number
    ori_name=sequences(i).name;
    ori_onlyName=ori_name(1:end-4);
    fprintf('The %d -th sequence: %s \n',i,ori_name);
    tempSplit=strsplit(ori_onlyName,'_');
    % rec_onlyName=[ori_onlyName,'_rec-',tempSplit{end}];
    rec_onlyName=[ori_onlyName,'_r02'];
    ori=pcread([ori_path,ori_name]);
    rec=pcread([rec_path,rec_onlyName,'.ply']);
    ori_loc=ori.Location;
    ori_color=ori.Color;
    rec_loc=rec.Location;
    numPoint=length(ori_loc);
    kdtreeObj_ori=KDTreeSearcher(ori_loc,'distance','euclidean');
    [idxnn_ori,dis]=knnsearch(kdtreeObj_ori,rec_loc,'k',1);
%     assert(length(find(dis~=0))==0);
    reorder_loc=ori_loc(idxnn_ori,:);
    reorder_color=ori_color(idxnn_ori,:);
%{
    h2=waitbar(0,'match the data');
    for j=1:numPoint
        str2=['matching...',num2str(j/numPoint),'%'];
        waitbar(j/numPoint,h2,str2);
        curlocation=rec_loc(j,:);
        idx=find(ismember(ori_loc,curlocation,'rows')==1);
        assert(length(idx)==1);
        reorder_loc(j,:)=ori_loc(idx,:);
        reorder_color(j,:)=ori.Color(idx,:);
    end
%}
    reorder_pt=pointCloud(reorder_loc, 'Color',reorder_color);
    pcwrite(reorder_pt,[reordered_recPath,ori_onlyName,'.ply']);
end
