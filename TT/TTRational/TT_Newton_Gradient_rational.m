function [V,dUx,y] = TT_Newton_Gradient_rational(A,Cb,U,y,residual,beta,dx_old,lambda)
% To solve (Ax)./(1+Cy) = b, we find Ax - (C.*b)y = b. Each iteration we
% update x and y by solving dx - (C.*b)dy = r

[d,m,r] = TTsizes(U);


%Right Orthogonalize, find every non-orthogal Ux to compute the seach
%directions dUx
[Ux,V] = TTmuOrthogonalizeRL(U);



[~,yl] = Ax_left(A,U,d);
[~,yr] = Ax_right(A,V,1);
[n_samples,~] = size(A{1});

dUx = cell(d,1);

%solve the seach directions
% for i = 1:d
%     for j = 1:m(i)
%         temp = residual.*A{i}(:,j);
%         temp = yr{i}.*repelem(temp,1,r(i+1));
%         dUx{i}((j-1)*r(i)+1:j*r(i),:) = (temp'*yl{i})';        
%         % dUx{i}((j-1)*r(i)+1:j*r(i),:) = (yr{i}'* diag(residual.*reshape(A(i,j,:),n_samples,1))*yl{i})';
%     end
% end

for i = 1:d
    dUx{i} = TTcore_Newton(yl{i},yr{i},A{i},Ux{i},residual,lambda);
end




%solve A*dx for computating step sizes
Adx = TTmuOrthogonalAx(A,dUx,yl,yr,m,d,r);

lambda2 = 0.1*lambda;
if beta <= 0
    %no momentum
    Adx = [Adx -Cb];
    %find Gram matrix with regularization
    reg_matrix = TTRegMatrix(dUx,U,V,lambda);
    reg_matrix = [reg_matrix zeros(d,d);zeros(d,d) lambda2^2*eye(d)];

    reg_vec = zeros(2*d,1);
    for i = 1:d
        reg_vec(i) = -lambda^2*sum(conj(dUx{i}).* Ux{i},"all");
    end
    reg_vec(d+1:2*d) = -lambda2^2*y;

    %solve step sizes, update seach directions
    alpha = (Adx'*Adx+reg_matrix)\(Adx'*residual+reg_vec);
    for i = 1:d
        dUx{i} = alpha(i)*dUx{i};
    end
    dy = alpha(d+1:2*d);
else
    %with momentum, project the search directions from the previous iteration
    dU_old = TT_Riemannian_projection(U,V,dx_old);
    Adx_old = TTmuOrthogonalAx(A,dU_old,yl,yr,m,d,r);
    Adx = [Adx Adx_old -Cb];

    %find Gram matrix with regularization
    reg_matrix = TTRegMatrix(dUx,U,V,lambda,dU_old);
    reg_matrix = [reg_matrix zeros(2*d,d);zeros(d,2*d) lambda2^2*eye(d)];
    reg_vec = zeros(3*d,1);

    for i = 1:d
        reg_vec(i) = -lambda^2*sum(conj(dUx{i}).* Ux{i},"all");
        reg_vec(i+d) = -lambda^2*sum(conj(dU_old{i}).* Ux{i},"all");
    end
    reg_vec(2*d+1:3*d) = -lambda2^2*y;
    %solve step sizes, update seach directions
    alpha = (Adx'*Adx+reg_matrix)\(Adx'*residual+reg_vec);
    for i = 1:d
        dUx{i} = alpha(i)*dUx{i}+alpha(i+d)*dU_old{i};
    end
    dUx{d} = dUx{d} + alpha(2*d+1)*Ux{d};
    dy = alpha(2*d+1:3*d);

end
y = y + dy;
end

