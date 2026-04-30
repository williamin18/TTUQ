function [x] = TT_rounding_ALS2(A,x,b,r_max,lambda)
%TT_ROUNDING_ALS Summary of this function goes here
%   Detailed explanation goes here

d = length(A);
[n_samples,~] = size(A{1});
[~,m,r] = TTsizes(x);

x = TTorthogonalizeRL(x); 

[~,yr] = Ax_right(A,x,1);      
yl = cell(d,1);
yl{1} = ones(n_samples,1);

for i = 1:d-1
    x{i} = TTcore_LS(yl{i},yr{i},A{i},b,lambda);
    %use SVD rounding

    
    %Use Erhard rounding
    xi = v2i(x{i},r(i));
    R = qr(A{i},'econ');
    Rxi = R*xi;
    Rxi = i2v(Rxi,r(i));
    [U,S,V] = svd(Rxi,'econ');
    r(i+1) = min(r(i+1),r_max);
    Rxi = U(:,1:r(i+1));
    Rxi = v2i(Rxi,r(i));
    xi = R\Rxi;
    xi = i2v(xi,r(i));
    [xi,R2] = qr(xi,'econ');
    x{i} = xi;

    Axi = A{i}*v2i(xi,r(i));
    Axi = reshape(Axi,[n_samples,r(i),r(i+1)]);
    yl{i+1} = zeros(n_samples,r(i+1));
    for j = 1:r(i+1)
        yl{i+1}(:,j)  = sum(yl{i}.*Axi(:,:,j),2);
    end

  
end
x{d} = TTcore_LS(yl{d},yr{d},A{d},b,lambda);
end