function [Y,y_m] = genHermiteMatrix(xi,order)
%GENLEGENDREMATRIX undefined
%   undefined
n = length(xi);
xi = reshape(xi,n,1);
Y = zeros(n,order);

Y(:,1) = ones(n,1);
Y(:,2) = xi;

for i = 2:order
    Y(:,i+1) = ( (xi.*Y(:,i)) - (i-1)*(Y(:,i-1)) );
end
y_m = Y(:,order);
end