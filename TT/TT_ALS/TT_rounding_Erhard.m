function x = TT_rounding_Erhard(A,x,r_max)
%TT_ROUNDING_ERHARD Summary of this function goes here
%   Detailed explanation goes here
d = length(A);
[n_samples,~] = size(A{1});
x = TTorthogonalizeRL(x); 
[~,m,r] = TTsizes(x);

yl = cell(d,1);
yl{1} = ones(n_samples,1);

x1 = x;
for i = 1:d-1
    Ayi = zeros(n_samples,r(i),m(i));
    for j = 1:m(i)
        Ayi(:,:,j) = yl{i}.*A{i}(:,j);
    end
    Ayi = reshape(Ayi,[n_samples r(i)*m(i)]);
    yli = Ayi*x{i};
    [U,S,V] = svd(yli,'econ');
    r(i+1) = min(r(i+1),r_max);
    Vr = V(:,1:r(i+1));

    [x{i},R] = qr(x{i}*Vr,'econ');
    x{i+1} = h2v(R*Vr'*v2h(x{i+1},m(i)),m(i));
    
    yl{i+1} = Ayi*x{i};
end
end

