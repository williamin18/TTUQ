function [V,dUx,y,pred,damping0] = TT_Newton_Gradient_rational_r1d2(A,C,U,y,residual,beta,dx_old,lambda,damping)
% To solve (Ax)./(1+Cy) = b, we find Ax ./ C.y = b. Each iteration we
% update x and y by solving dx*df/dx + dy*df/dy = r

[d,m,r] = TTsizes(U);


%Right Orthogonalize, find every non-orthogonal Ux to compute the search
%directions dUx
[Ux,V] = TTmuOrthogonalizeRL(U);



[~,yl] = Ax_left(A,U,d);
[~,yr] = Ax_right(A,V,1);
[n_samples,~] = size(A{1});

dUx = cell(d,1);

%TODO
Ax = sum((A{1}*Ux{1}).*yr{1},2);
[dfdy,Cy] = Ay_denominator(Ax,C,y);

%solve the search directions
for i = 1:d
    for j = 1:m(i)
        temp = residual.*A{i}(:,j);
        temp = yr{i}.*repelem(temp,1,r(i+1));
        dUx{i}((j-1)*r(i)+1:j*r(i),:) = (temp'*yl{i})';        
        % dUx{i}((j-1)*r(i)+1:j*r(i),:) = (yr{i}'* diag(residual.*reshape(A(i,j,:),n_samples,1))*yl{i})';
    end
end

% for i = 1:d
%     dUx{i} = TTcore_Newton(yl{i},yr{i},A{i},Ux{i},residual.*Cy,lambda);
% end




%solve A*dx for computing step sizes
dfdx = TTmuOrthogonalAx(A,dUx,yl,yr,m,d,r)./Cy;

lambda2 = 0.1*lambda;
if beta <= 0
    %no momentum
    dfdx = [dfdx dfdy];
    %find Gram matrix with regularization
    reg_matrix = TTRegMatrix(dUx,U,V,lambda);
    reg_matrix = [reg_matrix zeros(d,d);zeros(d,d) lambda2^2*eye(d)];

    reg_vec = zeros(2*d,1);
    for i = 1:d
        reg_vec(i) = -lambda^2*sum(conj(dUx{i}).* Ux{i},"all");
    end
    reg_vec(d+1:2*d) = -lambda2^2*y;
    Gram = dfdx'*dfdx;
    damping0 = max(abs(diag(Gram)));
    %solve step sizes, update search directions
    alpha = (Gram+damping0*eye(2*d)+reg_matrix)\(dfdx'*residual+reg_vec);
    for i = 1:d
        dUx{i} = alpha(i)*dUx{i};
    end
    dy = alpha(d+1:2*d);
else
    %with momentum, project the search directions from the previous iteration
    dU_old = TT_Riemannian_projection(U,V,dx_old);
    Adx_old = TTmuOrthogonalAx(A,dU_old,yl,yr,m,d,r)./Cy;
    dfdx = [dfdx Adx_old dfdy];

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
    Gram = dfdx'*dfdx+reg_matrix;
    alpha = (Gram+damping*eye(3*d))\(dfdx'*residual+reg_vec);
    for i = 1:d
        dUx{i} = alpha(i)*dUx{i}+alpha(i+d)*dU_old{i};
    end
    dy = alpha(2*d+1:3*d);

end
y = y + dy;
pred = 0.5*alpha'*Gram*alpha;
end

