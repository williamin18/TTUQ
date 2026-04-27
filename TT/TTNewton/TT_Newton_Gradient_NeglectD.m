function [V,dUx] = TT_Newton_Gradient_NeglectD(A,U,residual,beta,dx_old,lambda)
%V: right orthogonalization of U
%dUx: update of each TT-core

%A: left hand side matrix, rows in rank-1 format
%U: Unknown vector in TT format
%beta: if beta <= 0, no momentum
%dx_old: the update at the previous iteration
%lambda: regularization parameter, it is equal to sqrt(lambda)  in the paper
[d,m,r] = TTsizes(U);


%Right Orthogonalize, find every non-orthogal Ux to compute the seach
%directions dUx
[Ux,V] = TTmuOrthogonalizeRL(U);



[~,yl] = Ax_left(A,U,d);
[~,yr] = Ax_right(A,V,1);

dUx = cell(d,1);

%solve the seach directions
for i = 1:d-1
    dUx{i} = TTcore_Newton(yl{i},yr{i},A{i},Ux{i},residual,lambda);
end
dUx{d} = Ux{d};

% for i = 1:d
%     for j = 1:m(i)
%         temp = residual.*A{i}(:,j);
%         temp = yr{i}.*repelem(temp,1,r(i+1));
%         dUx{i}((j-1)*r(i)+1:j*r(i),:) = (temp'*yl{i})';        
%         % dUx{i}((j-1)*r(i)+1:j*r(i),:) = (yr{i}'* diag(residual.*reshape(A(i,j,:),n_samples,1))*yl{i})';
%     end
% end



%solve A*dx for computating step sizes
Adx = TTmuOrthogonalAx(A,dUx,yl,yr,m,d,r);

if beta <= 0
    %no momentum

    %find Gram matrix with regularization
    reg_matrix = TTRegMatrix(dUx,U,V,lambda);
    reg_vec = zeros(d,1);
    for i = 1:d
        reg_vec(i) = -lambda^2*sum(conj(dUx{i}).* Ux{i},"all");
    end
    
    %solve step sizes, update seach directions
    alpha = (Adx'*Adx+reg_matrix)\(Adx'*residual+reg_vec);
    for i = 1:d
        dUx{i} = alpha(i)*dUx{i};
    end
else
    %with momentum, project the search directions from the previous iteration
    dU_old = TT_Riemannian_projection(U,V,dx_old);
    Adx_old = TTmuOrthogonalAx(A,dU_old,yl,yr,m,d,r);
    Adx = [Adx Adx_old];

    %find Gram matrix with regularization
    reg_matrix = TTRegMatrix(dUx,U,V,lambda,dU_old);

    reg_vec = zeros(2*d,1);
    for i = 1:d
        reg_vec(i) = -lambda^2*sum(conj(dUx{i}).* Ux{i},"all");
        reg_vec(i+d) = -lambda^2*sum(conj(dU_old{i}).* Ux{i},"all");
    end

    %solve step sizes, update seach directions
    alpha = (Adx'*Adx+reg_matrix)\(Adx'*residual+reg_vec);
    for i = 1:d
        dUx{i} = alpha(i)*dUx{i}+alpha(i+d)*dU_old{i};
    end
end

end

