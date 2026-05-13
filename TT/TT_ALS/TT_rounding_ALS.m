function [x,Ax] = TT_rounding_ALS(A,x,b,r_max,lambda)
%TT_ROUNDING_ALS Summary of this function goes here
%   Detailed explanation goes here

d = length(A);
[n_samples,~] = size(A{1});
[~,m,r] = TTsizes(x);

x = TTorthogonalizeRL(x); 

[~,yr] = Ax_right(A,x,1);      
yl = cell(d,1);
yl{1} = ones(n_samples,1);
lambda2 = 0.1*lambda;

for i = 1:d-1
    xi = TTcore_LS(yl{i},yr{i},A{i},b,lambda);
    %use SVD rounding
    [U,~,~] = svd(xi,"econ");
    r(i+1) = min(r(i+1),r_max);
    x{i} = U(:,1:r(i+1));
    %Use Erhard rounding

    %-----------------------------------
    xi = reshape(x{i},[r(i), m(i), r(i+1)]);
    xi = reshape(permute(xi, [2 1 3]),m(i),[]);
    Axi = A{i}*xi;
    Axi = reshape(Axi,n_samples,r(i),r(i+1));

    yl{i+1} = zeros(n_samples,r(i+1));
    for j = 1:r(i+1)
        yl{i+1}(:,j)  = sum(yl{i}.*Axi(:,:,j),2);
    end
 
end
x{d} = TTcore_LS(yl{d},yr{d},A{d},b,lambda);
Ax = reshape(yl{d}.*permute(A{d},[1 3 2]),n_samples,[])*x{d};
end