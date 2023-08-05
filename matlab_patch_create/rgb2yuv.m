% convert the rgb component of the point cloud to yuv
% input is the rgb component of the point cloud, size: point_num x 3
% be always take care of the type of the data. uint8 or double?  
% uint8*-0.1=0
function yuv=rgb2yuv(rgb)
Point_Num=length(rgb);
rgb=double(rgb);
yuv=double(zeros(size(rgb)));
for i=1:Point_Num
    yuv(i,1)=0.2126*rgb(i,1)+0.7152*rgb(i,2)+0.0722*rgb(i,3);
    yuv(i,2)=-0.1146*rgb(i,1)-0.3854*rgb(i,2)+0.5000*rgb(i,3)+128;
    yuv(i,3)=0.5000*rgb(i,1)-0.4542*rgb(i,2)-0.0458*rgb(i,3)+128;
end
end
    
    
    
    
    
    
%   out_yuv[0] = float( ( 0.2126 * in_rgb[0] + 0.7152 * in_rgb[1] + 0.0722 * in_rgb[2]) / 255.0 );
%   out_yuv[1] = float( (-0.1146 * in_rgb[0] - 0.3854 * in_rgb[1] + 0.5000 * in_rgb[2]) / 255.0 + 0.5000 );
%   out_yuv[2] = float( ( 0.5000 * in_rgb[0] - 0.4542 * in_rgb[1] - 0.0458 * in_rgb[2]) / 255.0 + 0.5000 );   