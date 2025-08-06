function [sample_idx,sample_xi,n_samples2] = genTensorCompletionSamples(m,d,n_samples,polynomial)
%n_samples2 is usually the same as n_samples, only samller when there are
%too many repeated samples

switch polynomial
    case "Hermite"
        f = @genHermite;
        poly_integral = zeros(m+1,1);
        poly_integral(1) = sqrt(2*pi);
    otherwise
        err('Unsupported polynomial type')
end


xi_roots = genPolynomialRoots(m+1,polynomial);
H = zeros(m+1,m+1);
for i = 1:m+1
    H(i,:) = f(xi_roots,i-1)';
end
weights = H\poly_integral;

temp = 0;
split_points = zeros(m+1,1);
for i = 1:m+1
    temp = temp+weights(i);
    split_points(i) = temp;
end

lh_samples = split_points(m+1)*lhsdesign(2*n_samples,d);
sample_idx = zeros(2*n_samples,d);
prev_split = 0;
for i = 1:m+1
    temp = i*((lh_samples >= prev_split)&(lh_samples < split_points(i)));
    sample_idx = sample_idx+temp;
    prev_split =  split_points(i);
end

sample_idx = unique(sample_idx,'rows','stable');
[n_samples2,~] = size(sample_idx);
if n_samples2>n_samples
    sample_idx = sample_idx(1:n_samples,:);
    n_samples2 = n_samples;
end

sample_xi = xi_roots(sample_idx);




end

