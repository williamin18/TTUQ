function [y] = genPolynomialFreqSamplesTensor2(samples,order,polynomial,f_order)
%GENHERMITE Summary of this function goes here
%   Detailed explanation goes here


switch polynomial
    case "Hermite"
        f = @genHermiteMatrix;
    case "Legendre"
        f = @genLegendreMatrix;
    otherwise
        err('Unsupported polynomial type')
end

[n,d]=size(samples);

for i = 1:d-1
    y{i} = f(samples(:,i),order);
end

y{d}= genLegendreMatrix(samples(:,d),f_order);


end

