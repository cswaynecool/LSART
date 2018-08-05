function [x,x1,y,y1]=generate_displacement()

[x,y]=meshgrid(-5:5,-5:5);
x1=x.^2;
y1=y.^2;
x=permute(x,[2 1]);
y=permute(y,[2 1]);
x1=permute(x1,[2 1]);
y1=permute(y1,[2 1]);