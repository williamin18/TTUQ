function [y,Cy] = TT_rounding_ALS_denominator(Ax,C,y,b,r_max,lambda)
%TT_ROUNDING_ALS3 Summary of this function goes here
%   Detailed explanation goes here
d = length(C);
[n_samples,~] = size(C{1});

y = TTorthogonalizeRL(y); 
[~,m,r] = TTsizes(y);

[~,yr] = Ax_right(C,y,1);      
yl = cell(d,1);
yl{1} = ones(n_samples,1);

x1 = y;
for i = 1:d-1

    [~,~,V] = svd(y{i},'econ');
    r(i+1) = min(r(i+1),r_max);
    V = V(:,1:r(i+1));
    y{i+1} = h2v(V'*v2h(y{i+1},m(i+1)),m(i+1));
    
    
    Ay2 = zeros(n_samples,r(i+2),m(i+1));
    for j = 1:m(i+1)
        Ay2(:,:,j) = yr{i+1}.*C{i+1}(:,j); 
    end
    Ay2 = reshape(permute(Ay2,[3 2 1]),[m(i+1)*r(i+2) n_samples]);
    yri = (v2h(y{i+1},m(i+1))*Ay2).';
    
    J = (yl{i}(-Ax./(Cy.^2)) ).*permute(A{i},[1 3 2]).*permute(yri{i},[1 3 4 2]);

    y{i} = TTcore_LS(yl{i},yri,C{i},b,lambda);
    [y{i},R] = qr(y{i},'econ');
    y{i+1} = h2v(R*v2h(y{i+1},m(i+1)),m(i+1));


    Axi = C{i}*v2i(y{i},r(i));
    Axi = reshape(Axi,[n_samples,r(i),r(i+1)]);
    yl{i+1} = zeros(n_samples,r(i+1));
    for j = 1:r(i+1)
        yl{i+1}(:,j) = sum(yl{i}.*Axi(:,:,j),2);
    end

  
end
y{d} = TTcore_LS(yl{d},yr{d},C{d},b,lambda);
Ax = reshape(yl{d}.*permute(C{d},[1 3 2]),n_samples,[])*y{d};
end

