function [Y,y_m] = genShiftedLegendreMatrix(xi,order)
% Basis for xi uniformly distributed between 0 and 1
%   undefined
n = length(xi);
xi = reshape(xi,n,1);
Y = zeros(n,order);

Y(:,1) = ones(n,1);
Y(:,2) = 2*xi-1;

for i = 2:order
    Y(:,i+1) = ( (2*i-1)*((2*xi-1).*Y(:,i)) - (i-1)*(Y(:,i-1)) )/i;
end
y_m = Y(:,order);
end