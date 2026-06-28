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
    case 7 
        y = 1/16*(429*xi^7-693*xi^5+315*xi^3-35*xi);
    case 8
        y = 1/128*(6435*xi^8 - 12012*xi^6 + 6930*xi^4 - 1260*xi^2 + 35);
    otherwise
        err('Unsupported order')

end
end

