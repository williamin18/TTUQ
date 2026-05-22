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
        y = 1/8*(35*xi.^4 - 30*xi.^2+3);
    case 5
        y = 1/8*(63*xi.^5 - 70*xi.^3 + 15*xi);
    case 6
        y = 1/16*(231*xi^6-315*xi^4+105*xi^2-5);
    otherwise
        err('Unsupported order')

end
end

