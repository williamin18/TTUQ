function [x,y,Ax,Cy] = TT_rounding_ALS_RPC(A,x,C,y,b,r_max,lambda)
%TT_ROUNDING_ALS Summary of this function goes here
%   Detailed explanation goes here

x = TTorthogonalizeRL(x); 
y = TTorthogonalizeRL(y); 

d = length(A);
[n_samples,~] = size(A{1});
[d,m,r] = TTsizes(x);
[~,my,ry] = TTsizes(y);

[~,axr] = Ax_right(A,x,1);
[~,cyr] = Ax_right(C,y,1);

axl = cell(d,1);
axl{1} = ones(n_samples,1);
cyl = cell(d,1);
cyl{1} = ones(n_samples,1);

Ax = sum((A{1}*x{1}).*axr{1},2);

for i = 1:d-1
    ni = ry(i)*my(i)*ry(i+1);
    Ci = cyl{i}.*permute(C{i},[1 3 2]).*permute(cyr{i},[1 3 4 2]);
    Ci = reshape(Ci, n_samples,ni);
    Cy = Ci*reshape(y{i},[],1);
    residual = Ax./(1+Cy)-b;
    
    Jyi = (-Ax./(Cy.^2)).*Ci;
    Jy_reg = [Jyi;lambda*eye(ni)];
    res_reg = [residual; -lambda*reshape(y{i},[],1)];
    dYi = Jy_reg\res_reg;
    yi = y{i}+reshape(dYi,[ry(i)*my(i) ry(i+1)]);

    
    [U,S,V] = svd(yi,"econ");
    ry(i+1) = min(ry(i+1),r_max);
    y{i} = U(:,1:ry(i+1));

    y{i+1} = h2v(S(1:ry(i+1),1:ry(i+1))*V(:,1:ry(i+1))'*v2h(y{i+1},my(i+1)),my(i+1));
    cyl{i+1} = reshape(cyl{i}.*permute(C{i},[1 3 2]),n_samples,[])*y{i};

end
i = d;
ni = ry(d)*my(i)*ry(i+1);
Ci = cyl{i}.*permute(C{i},[1 3 2]).*permute(cyr{i},[1 3 4 2]);
Ci = reshape(Ci, n_samples,ni);
Cy = Ci*y{i};
residual = Ax./(1+Cy)-b;

Jyi = (-Ax./(Cy.^2)).*Ci;
Jy_reg = [Jyi;lambda*eye(ni)];
res_reg = [residual; -lambda*reshape(y{i},[],1)];
dYi = Jy_reg\res_reg;
y{d} = y{i}+dYi;

Cy = 1+Ci*y{d};

for i = 1:d-1
    ni = r(i)*m(i)*r(i+1);
    Jxi = (axl{i}./Cy ).*permute(A{i},[1 3 2]).*permute(axr{i},[1 3 4 2]);
    Jxi = reshape(Jxi,n_samples,ni);
    Jx_reg = [Jxi ;lambda*eye(ni)];
    b_reg = [b; zeros(ni,1)];
    xi = reshape(Jx_reg\b_reg,[r(i)*m(i) r(i+1)]);


    [U,S,V] = svd(xi,"econ");
    r(i+1) = min(r(i+1),r_max);
    x{i} = U(:,1:r(i+1));

    axl{i+1} = reshape(axl{i}.*permute(A{i},[1 3 2]),n_samples,[])*x{i};
end
x{d} =  h2v(S(1:r(d),1:r(d))*V(:,1:r(d))'*v2h(x{d},m(d)),m(d));

Ax = reshape(axl{d}.*permute(A{d},[1 3 2]),n_samples,[])*x{d};
Cy = Cy-1;
end