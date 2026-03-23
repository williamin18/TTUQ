function [Xk,y] = TTcore_LS_RPC(yl,yr,Ak,b,Cb,d,lambda,lambda2)
%TTCORE_LS Summary of this function goes here
%   Detailed explanation goes here
[~,m] = size(Ak);
[~,r_k] = size(yl);
[~,r_k1] = size(yr);

Y_yl = kron( kron(ones(1,r_k1), ones(1,m)), yl );
Y_Ai = kron( kron(ones(1,r_k1), Ak ),       ones(1,r_k));
Y_yr = kron( kron(yr,           ones(1,m)), ones(1,r_k));

Y = Y_yl.*Y_Ai.*Y_yr;

n = r_k*m*r_k1;

Y = [Y -Cb];
%regularization
if lambda ~= 0
    Y = [Y;lambda*eye(n),zeros(n,d)];
    b = [b; zeros(n,1)];
end
if lambda2~= 0
    Y = [Y;zeros(d,n),lambda2*eye(d)];
    b = [b; zeros(d,1)];
end
Xky = Y\b;
Xk = reshape(Xky(1:n),r_k*m,r_k1);
y = Xky(n+1:end);

end