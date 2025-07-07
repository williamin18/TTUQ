function [y] = genLegendre(xi,order)
%GENHERMITE Summary of this function goes here
switch order
    case 0
        y = ones(size(xi));
    case 1
        y = xi;
    case 2
        y = 0.5*(3*xi.^2 - 1);
    case 3
        y = 0.5*(5*xi.^3 - 3*xi);
    case 4
        y = 1/8*(35*xi.^4 - 30*xi.^3+3);
    case 5
        y = 1/8*(63*xi.^5 - 70*xi.^3 + 15*xi);
    otherwise
        err('Unsupported order')

end
end

