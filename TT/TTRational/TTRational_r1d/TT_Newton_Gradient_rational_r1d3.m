function [V,dU,Adx,W,dY,Cdy] = TT_Newton_Gradient_rational_r1d3(A,C,U,Y,residual,beta,dx_old,dy_old,lambda)
% To solve (Ax)./(1+Cy) = b, we find Ax ./ C.y = b. Each iteration we
% update x and y by solving dx*df/dx + dy*df/dy = r




%Right Orthogonalize, find every non-orthogonal Ux to compute the search
%directions dUx
[Ux,V] = TTmuOrthogonalizeRL(U);
[Yy,W] = TTmuOrthogonalizeRL(Y);

[d,m,r] = TTsizes(U);
[~,my,ry] = TTsizes(Y);


[~,axl] = Ax_left(A,U,d);
[~,axr] = Ax_right(A,V,1);
[n_samples,~] = size(A{1});

[~,cyl] = Ax_left(C,Y,d);
[~,cyr] = Ax_right(C,W,1);


dU = cell(d,1);
dY = cell(d,1);

Ax = sum((A{1}*Ux{1}).*axr{1},2);
Cy = sum((C{1}*Yy{1}).*cyr{1},2)+1;



Jx = cell(d,1);
Jy = cell(d,1);
%solve the search directions
for i = 1:d
    ni = r(i)*m(i)*r(i+1);
    Jxi = (axl{i}./Cy ).*permute(A{i},[1 3 2]).*permute(axr{i},[1 3 4 2]);
    Jx{i} = reshape(Jxi,n_samples,ni);
    Jx_reg = [Jx{i} ;lambda*eye(ni)];
    res_reg = [residual; -lambda*reshape(Ux{i},[],1)];
    dUi = Jx_reg\res_reg;
    dU{i} = reshape(dUi,[r(i) m(i) r(i+1)]);


    ni = ry(i)*my(i)*ry(i+1);
    Jyi = (cyl{i}.*(-Ax./(Cy.^2)) ).*permute(C{i},[1 3 2]).*permute(cyr{i},[1 3 4 2]);
    Jy{i} = reshape(Jyi,n_samples,ni);
    Jy_reg = [Jy{i};lambda*eye(ni)];
    res_reg = [residual; -lambda*reshape(Yy{i},[],1)];
    dYi = Jy_reg\res_reg;
    dY{i} = reshape(dYi,[r(i) m(i) r(i+1)]);
end

dfdx = zeros(n_samples,d);
dfdy = zeros(n_samples,d);
for i = 1:d
    dfdx(:,i) = Jx{i}*reshape(dU{i},[],1);
    dfdy(:,i) = Jy{i}*reshape(dY{i},[],1);
end

if beta <= 0
    %no momentum
    Ja = [dfdx dfdy];
    %find Gram matrix with regularization
    reg_matrix = [TTRegMatrix(dU,U,V,lambda) zeros(d,d);zeros(d,d) TTRegMatrix(dY,Y,W,lambda)];
    reg_vec = zeros(2*d,1);
    for i = 1:d
        reg_vec(i) = -lambda^2*sum(conj(dU{i}).* Ux{i},"all");
        reg_vec(d+i) = -lambda^2*sum(conj(dY{i}).* Yy{i},"all");
    end

    %solve step sizes, update search directions
    alpha = (Ja'*Ja+reg_matrix)\(Ja'*residual+reg_vec);
    for i = 1:d
        dU{i} = alpha(i)*dU{i};
        dY{i} = alpha(d+i)*dY{i};
    end


else
    %with momentum, project the search directions from the previous iteration
    dU_old = TT_Riemannian_projection(U,V,dx_old);
    dY_old = TT_Riemannian_projection(Y,W,dy_old);
    dfdx2 = zeros(n_samples,d);
    dfdy2 = zeros(n_samples,d);
    for i = 1:d
        dfdx2(:,i) = Jx{i}*reshape(dU_old{i},[],1);
        dfdy2(:,i) = Jy{i}*reshape(dY_old{i},[],1);
    end


    Ja = [dfdx dfdx2 dfdy dfdy2];

    %find Gram matrix with regularization
    reg_matrix = [TTRegMatrix(dU,U,V,lambda,dU_old) zeros(2*d,2*d);zeros(2*d,2*d) TTRegMatrix(dY,Y,W,lambda,dY_old)];
    reg_vec = zeros(4*d,1);

    for i = 1:d
        reg_vec(i) = -lambda^2*sum(conj(dU{i}).* Ux{i},"all");
        reg_vec(d+i) = -lambda^2*sum(conj(dU_old{i}).* Ux{i},"all");
        reg_vec(2*d+i) =  -lambda^2*sum(conj(dY{i}).* Yy{i},"all");
        reg_vec(3*d+i) = -lambda^2*sum(conj(dY_old{i}).* Yy{i},"all");
    end
    %solve step sizes, update seach directions
    alpha = (Ja'*Ja+reg_matrix)\(Ja'*residual+reg_vec);
    for i = 1:d
        dU{i} = alpha(i)*dU{i}+alpha(d+i)*dU_old{i};
        dY{i} = alpha(2*d+i)*dY{i}+alpha(3*d+i)*dY_old{i};
    end


end

end

