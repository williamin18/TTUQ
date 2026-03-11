function [x] = TT_rounding_ALS4(A,x,b,r_max,lambda)
%TT_ROUNDING_ALS3 Summary of this function goes here
%   Detailed explanation goes here
d = length(A);
[n_samples,~] = size(A{1});
[~,m,r] = TTsizes(x);

x = TTorthogonalizeRL(x); 

[~,yr] = Ax_right(A,x,1);      
yl = cell(d,1);
yl{1} = ones(n_samples,1);

for i = 1:d-1

    [~,~,V] = svd(v2h(x{i+1},m(i+1)));
    r(i+1) = min(r(i+1),r_max);
    
    yri = U(:,1:r(i+1));
    x{i} = TTcore_LS(yl{i},yri,A{i},b,lambda);
    [x{i},~] = qr(x{i},'econ');
    


    Axi = A{i}*v2i(x{i},r(i));
    Axi = reshape(Axi,[n_samples,r(i),r(i+1)]);
    yl{i+1} = zeros(n_samples,r(i+1));
    for j = 1:r(i+1)
        yl{i+1}(:,j)  = sum(yl{i}.*Axi(:,:,j),2);
    end

  
end
x{d} = TTcore_LS(yl{d},yr{d},A{d},b,lambda);
end

