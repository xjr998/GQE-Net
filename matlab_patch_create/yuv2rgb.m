% convert the rgb component of the point cloud to yuv
% input is the rgb component of the point cloud, size: point_num x 3
% be always take care of the type of the data. uint8 or double?  
% uint8*-0.1=0
function rgb=yuv2rgb(yuv)
Point_Num=length(yuv);
yuv=double(yuv);
rgb=double(zeros(size(yuv)));
for i=1:Point_Num
    rgb(i,1)=yuv(i,1)+1.5748*(yuv(i,3)-128);
    rgb(i,2)=yuv(i,1)-0.1881*(yuv(i,2)-128)-0.4681 *(yuv(i,3)-128);
    rgb(i,3)=yuv(i,1)+1.8556*(yuv(i,2)-128);
end
end
    
    
    
    
    
    
%   out_yuv[0] = float( ( 0.2126 * in_rgb[0] + 0.7152 * in_rgb[1] + 0.0722 * in_rgb[2]) / 255.0 );
%   out_yuv[1] = float( (-0.1146 * in_rgb[0] - 0.3854 * in_rgb[1] + 0.5000 * in_rgb[2]) / 255.0 + 0.5000 );
%   out_yuv[2] = float( ( 0.5000 * in_rgb[0] - 0.4542 * in_rgb[1] - 0.0458 * in_rgb[2]) / 255.0 + 0.5000 );   