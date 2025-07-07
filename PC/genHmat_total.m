function [H] = genHmat_total(M,d)
%GENHMAT_TOTAL Summary of this function goes here
%   This function generates the order of every combination of univariate
%   polynomials for a mutilvarite polynomial recursively
%   M is order of polynomial, d is number of parameters. In total order,
%   the order of multi-variate polynomial is up to M

H = zeros(0,d);

if(d ==0)
    H = zeros(1,0);
elseif(M==0)
    H = zeros(1,d);
else
    for i = 0:M
        Hi = genHmat_total(M-i,d-1);
        [m,~] = size(Hi);
        Hi1 = i*ones(m,1);
        Hi = [Hi1 Hi];
        H = [H;Hi];
    end
end

end

