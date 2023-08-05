function DATA=rectangular(input,n)
% 将数据进行“回字形”排列，输入是1024x3，输出是以“回字形”排列好的32x32x3的数据
% n=8;
% input=rand(64,3);
upb=1; % upper bound
lob=n; % lower bound
lfb=1;  % left bound
rtb=n;  % right bound
[pointNum,~]=size(input);     % 输入的点的个数，input是按照距离代表点由近及远的顺序排列
j=1;
i=0;
data=zeros(n,n,3);
data_y=zeros(n,n,1);
r=zeros(n,n);
g=zeros(n,n);
b=zeros(n,n);
while(pointNum>0)
    while(i<lob)
        i=i+1;
        r(i,j)=input(pointNum,1);
%         g(i,j)=input(pointNum,2);
%         b(i,j)=input(pointNum,3);   % i=32
        pointNum=pointNum-1;
    end
    lob=lob-1;
    while(j<rtb)
        j=j+1;
        r(i,j)=input(pointNum,1);
%         g(i,j)=input(pointNum,2);
%         b(i,j)=input(pointNum,3);
        pointNum=pointNum-1;
    end
    rtb=rtb-1;
    while(i>upb)
        i=i-1;
        r(i,j)=input(pointNum,1);
%         g(i,j)=input(pointNum,2);
%         b(i,j)=input(pointNum,3);
        pointNum=pointNum-1;
    end
    upb=upb+1;
    while(j>lfb+1)
        j=j-1;
        r(i,j)=input(pointNum,1);
%         g(i,j)=input(pointNum,2);
%         b(i,j)=input(pointNum,3);
        pointNum=pointNum-1;
    end
    lfb=lfb+1;
end
data_y(:,:,1)=r;
% data(:,:,2)=g;
% data(:,:,3)=b;
DATA=data_y;