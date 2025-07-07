function [yrk,yr] = Ax_right(A,x,k)
%multiply rank 1 samples A with TT x for cores indices greater than k, 
% yr{k} is a n*r_k matrix: n is number of samples, r_k is the rank of k's TT-core

[n_samples,~] = size(A{1});
[d,m,r] = TTsizes(x);

yl = cell(d-k+1,1);
yr{d-k+1} = ones(n_samples,1);

for i = d:-1:k+1
    xi = reshape(x{i},[r(i), m(i), r(i+1)]);
    xi = reshape(permute(xi, [2 3 1]),m(i),[]);
    Axi = A{i}*xi;
    Axi = reshape(Axi,n_samples,r(i+1),r(i));
    temp = zeros(n_samples,r(i));
    for j = 1:r(i)
        temp(:,j) = sum(yr{i-k+1}.*Axi(:,:,j),2);
    end
    yr{i-k} = temp;
end
yrk = yr{1};
end

