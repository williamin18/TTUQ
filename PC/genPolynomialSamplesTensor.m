function [y] = genPolynomialSamplesTensor(samples,order,polynomial)
%GENHERMITE Summary of this function goes here
%   Detailed explanation goes here


switch polynomial
    case "Hermite"
        f = @genHermite;
    case "Legendre"
        f = @genLegendre;
    otherwise
        err('Unsupported polynomial type')
end
[n,d]=size(samples);
y = cell(d,1);
for i = 1:d
    for j = 0:order
        y{i}(:,j+1) = f(samples(:,i),j);
    end
end

end

