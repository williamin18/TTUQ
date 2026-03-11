function [x] = TT_rounding_ALS3(A,x,b,r_max,lambda)
%TT_ROUNDING_ALS3 Summary of this function goes here
%   Detailed explanation goes here
d = length(A);
[n_samples,~] = size(A{1});
x = TTorthogonalizeRL(x); 
[~,m,r] = TTsizes(x);

[~,yr] = Ax_right(A,x,1);      
yl = cell(d,1);
yl{1} = ones(n_samples,1);

for i = 1:d-1


    
    %Use Erhard rounding to approximate (i+1)-th TT-core
    Ay2 = zeros(n_samples,r(i+2),m(i+1));
    for j = 1:m(i+1)
        Ay2(:,:,j) = yr{i+1}.*A{i+1}(:,j);
    end
    Ay2 = reshape(permute(Ay2,[3 2 1]),[m(i+1)*r(i+2) n_samples]);
    [~,R] = qr(Ay2','econ');

    xi2 = v2h(x{i+1},m(i+1));
    yri = xi2*R';
    [~,~,V] = svd(yri,'econ');
    r(i+1) = min(r(i+1),r_max);
    V = V(:,1:r(i+1))'/R;
    [Q, ~] = qr(V', 'econ');
    x2 = Q';
    yri = (x2*Ay2).';
    x{i} = TTcore_LS(yl{i},yri,A{i},b,lambda);
    [x{i},~] = qr(x{i},'econ');
    

    % [U,~,~] = svd(yr{i},'econ');
    % r(i+1) = min(r(i+1),r_max);
    % yri = U(:,1:r(i+1));
    % x{i} = TTcore_LS(yl{i},yri,A{i},b,lambda);
    % [x{i},~] = qr(x{i},'econ');
    


    Axi = A{i}*v2i(x{i},r(i));
    Axi = reshape(Axi,[n_samples,r(i),r(i+1)]);
    yl{i+1} = zeros(n_samples,r(i+1));
    for j = 1:r(i+1)
        yl{i+1}(:,j)  = sum(yl{i}.*Axi(:,:,j),2);
    end

  
end
x{d} = TTcore_LS(yl{d},yr{d},A{d},b,lambda);
end

