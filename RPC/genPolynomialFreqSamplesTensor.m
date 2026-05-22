function [y,f_scale] = genPolynomialFreqSamplesTensor(samples,order,polynomial,freq,f_order)
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
n_freq = length(freq);

y = cell(d+1,1);
for i = 1:d
    for j = 0:order
        y{i}(:,j+1) = kron(ones(n_freq),f(samples(:,i),j));
    end
end


f_scale = max(freq);
freq = reshape(freq/freq_scale,n_freq,1);

for j = 1:f_order
    y{d+1}(:,j+1) = kron(genLegendre(freq,j),ones(n,1));
end

end

