function [y] = genHermite(xi,order)
%GENHERMITE Summary of this function goes here
switch order
    case 0
        y = ones(size(xi));
    case 1
        y = xi;
    case 2
        y = xi.^2 - 1;
    case 3
        y = xi.^3 - 3*xi;
    case 4
        y = xi.^4 - 6*xi.^2+3;
    case 5
        y = xi.^5 - 10*xi.^3 + 15*xi;
    otherwise
        err('Unsupported order')
end
end

