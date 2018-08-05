function [location] = region2location(I, region, varargin)

gray = double(rgb2gray(I));

[height, width] = size(gray);

% If the provided region is a polygon ...
if numel(region) > 4
    num = numel(region)/4;
    %         x1 = round(min(region(1:2:end)));
    %         x2 = round(max(region(1:2:end)));
    %         y1 = round(min(region(2:2:end)));
    %         y2 = round(max(region(2:2:end)));
    x1 = round(sort(region(1:2:end), 'ascend'));
    x1 = round(mean(x1(1:num)));
    x2 = round(sort(region(1:2:end), 'descend'));
    x2 = round(mean(x2(1:num)));
    y1 = round(sort(region(2:2:end), 'ascend'));
    y1 = round(mean(y1(1:num)));
    y2 = round(sort(region(2:2:end), 'descend'));
    y2 = round(mean(y2(1:num)));
    region2location.m
    region = round([x1, y1, x2 - x1, y2 - y1]);
else
    region = round([round(region(1)), round(region(2)), ...
        round(region(1) + region(3)) - round(region(1)), ...
        round(region(2) + region(4)) - round(region(2))]);
end;

x1 = max(0, region(1));
y1 = max(0, region(2));
x2 = min(width-1, region(1) + region(3) - 1);
y2 = min(height-1, region(2) + region(4) - 1);


location = [x1, y1, x2 - x1 + 1, y2 - y1 + 1];
