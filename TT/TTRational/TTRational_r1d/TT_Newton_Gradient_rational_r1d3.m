function [U,V,dU,Adx,Y,W,dY,Cdy] = TT_Newton_Gradient_rational_r1d3(A,C,U,Y,b,residual,beta,dx_old,dy_old,lambda)
% To solve (Ax)./(1+Cy) = b, we find Ax ./ C.y = b. Each iteration we
% update x and y by solving dx*df/dx + dy*df/dy = r

[d,m,r] = TTsizes(U);
[~,my,ry] = TTsizes(Y);
[n_samples,~] = size(A{1});

%Output update directions
dU = cell(d,1);
dY = cell(d,1);

%Store Jacobian for each update direction
Jx = cell(d,1);
Jy = cell(d,1);
lambda2 = 1*lambda;

%Left contraction
[~,axl] = Ax_left(A,U,d);
[~,cyl] = Ax_left(C,Y,d);

% %Least squares retraction for the last TT-core of denominator
% Ad = reshape(axl{d}.*permute(A{d},[1 3 2]),n_samples,[]);
% Cd = reshape(cyl{d}.*permute(C{d},[1 3 2]),n_samples,[]);
% Ax = Ad*U{d};
% Cy = Cd*Y{d}+1;
% 
% ni = ry(d)*my(d);
% Jyd = (-Ax./Cy.^2).*Cd;
% Jy_reg = [Jyd;lambda2*eye(ni)];
% res_reg = [residual; -lambda2*reshape(Y{d},ni,1)];
% dY{d} = Jy_reg\res_reg;
% Y{d} = Y{d}+dY{d};
% 
% Cy = Cd*Y{d}+1;
% residual = b-Ax./Cy;
% 
% %Least squares retraction for the last TT-core of numerator
% ni = r(d)*m(d);
% Jx{d} = Ad./Cy;
% Jx_reg = [Jx{d};lambda*eye(ni)];
% res_reg = [residual; -lambda*reshape(U{d},ni,1)];
% dU{d} = Jx_reg\res_reg;
% U{d} = U{d}+dU{d};
% 
% Ax = Ad*U{d};
% residual = b - Ax./Cy;

% Jy{d} = (-Ax./Cy.^2).*Cd;


%Right Orthogonalize, find every non-orthogonal Ux to compute the search
%directions dUx
[Ux,V] = TTmuOrthogonalizeRL(U);
[Yy,W] = TTmuOrthogonalizeRL(Y);

[~,axr] = Ax_right(A,V,1);
[~,cyr] = Ax_right(C,W,1);


Ax = sum((A{1}*Ux{1}).*axr{1},2);
Cy = sum((C{1}*Yy{1}).*cyr{1},2)+1;


%solve the search directions
for i = 1:d
    ni = r(i)*m(i)*r(i+1);
    Jxi = (axl{i}./Cy ).*permute(A{i},[1 3 2]).*permute(axr{i},[1 3 4 2]);
    Jx{i} = reshape(Jxi,n_samples,ni);
    Jx_reg = [Jx{i} ;lambda*eye(ni)];
    res_reg = [residual; -lambda*reshape(Ux{i},ni,1)];
    dUi = Jx_reg'*res_reg;
    dU{i} = reshape(dUi,[r(i)*m(i) r(i+1)]);


    ni = ry(i)*my(i)*ry(i+1);
    Jyi = (cyl{i}.*(-Ax./(Cy.^2)) ).*permute(C{i},[1 3 2]).*permute(cyr{i},[1 3 4 2]);
    Jy{i} = reshape(Jyi,n_samples,ni);
    Jy_reg = [Jy{i};lambda2*eye(ni)];
    res_reg = [residual; -lambda2*reshape(Yy{i},ni,1)];
    dYi = Jy_reg'*res_reg;
    dY{i} = reshape(dYi,[r(i)*m(i) r(i+1)]);
end

% i = d;
% ni = ry(i)*my(i)*ry(i+1);
% Jyi = (cyl{i}.*(-Ax./(Cy.^2)) ).*permute(C{i},[1 3 2]).*permute(cyr{i},[1 3 4 2]);
% Jy{i} = reshape(Jyi,n_samples,ni);
% Jy_reg = [Jy{i};lambda2*eye(ni)];
% res_reg = [residual; -lambda2*reshape(Yy{i},ni,1)];
% dYi = Jy_reg'*res_reg;
% dY{i} = reshape(dYi,[r(i)*m(i) r(i+1)]);

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
    reg_matrix = [TTRegMatrix(dU,U,V,lambda) zeros(d,d);zeros(d,d) TTRegMatrix(dY,Y,W,lambda2)];
    reg_vec = zeros(2*d,1);
    for i = 1:d
        reg_vec(i) = -lambda^2*sum(conj(dU{i}).* Ux{i},"all");
        reg_vec(d+i) = -lambda2^2*sum(conj(dY{i}).* Yy{i},"all");
    end

    %solve step sizes, update search directions
    alpha = (Ja'*Ja+reg_matrix)\(Ja'*residual+reg_vec);
    for i = 1:d
        dU{i} = alpha(i)*dU{i};
        dY{i} = alpha(d+i)*dY{i};
    end

    Adx = dfdx*alpha(1:d).*Cy;
    Cdy = dfdy*alpha(d+1:2*d).*(-Cy.^2./Ax);

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
    reg_matrix = [TTRegMatrix(dU,U,V,lambda,dU_old) zeros(2*d,2*d);zeros(2*d,2*d) TTRegMatrix(dY,Y,W,lambda2,dY_old)];
    reg_vec = zeros(4*d,1);

    for i = 1:d
        reg_vec(i) = -lambda^2*sum(conj(dU{i}).* Ux{i},"all");
        reg_vec(d+i) = -lambda^2*sum(conj(dU_old{i}).* Ux{i},"all");
        reg_vec(2*d+i) =  -lambda2^2*sum(conj(dY{i}).* Yy{i},"all");
        reg_vec(3*d+i) = -lambda2^2*sum(conj(dY_old{i}).* Yy{i},"all");
    end
    %solve step sizes, update search directions
    alpha = (Ja'*Ja+reg_matrix)\(Ja'*residual+reg_vec);
    for i = 1:d
        dU{i} = alpha(i)*dU{i}+alpha(d+i)*dU_old{i};
        dY{i} = alpha(2*d+i)*dY{i}+alpha(3*d+i)*dY_old{i};
    end

    Adx = [dfdx dfdx2]*alpha(1:2*d).*Cy;
    Cdy = [dfdy dfdy2]*alpha(2*d+1:4*d).*(-Cy.^2./Ax);
end


% Adx2 = zeros(n_samples);
% Cdy2 = zeros(n_samples);
% for i = 1:d
%     Jxi = axl{i}.*permute(A{i},[1 3 2]).*permute(axr{i},[1 3 4 2]);
%     Jxi = reshape(Jxi,n_samples,[]);
%     Adx2 = Adx2+Jxi*reshape(dU{i},[],1);
% 
%     Jyi = cyl{i}.*permute(C{i},[1 3 2]).*permute(cyr{i},[1 3 4 2]);
%     Jyi = reshape(Jyi,n_samples,[]);
%     Cdy2 = Cdy2+Jyi*reshape(dY{i},[],1);
% end



end

