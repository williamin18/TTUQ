function [x,y] = TT_rounding_ALS_RPC(A,x,b,Cb,y,r_max,lambda)
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
    xi = TTcore_LS_RPC(yl{i},yr{i},A{i},b,Cb,d,lambda,lambda2);
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
x{d} = TTcore_LS_RPC(yl{d},yr{d},A{d},b,Cb,d,lambda,lambda2);
end