function [V,dUx] = TT_Newton_Gradient(A,U,residual,beta,dx_old,lambda)
%V: right orthogonalization of U
%dUx: update of each TT-core

%A: left hand side matrix, rows in rank-1 format
%U: Unknown vector in TT format
%beta: if beta <= 0, no momentum
%dx_old: the update at the previous iteration
%lambda: regularization parameter, it is equal to sqrt(lambda)  in the paper
[n_samples,~] = size(A{1});
[d,m,r] = TTsizes(U);
V = U;
Ux = cell(d,1);
Ux{d} = U{d};


%Right Orthogonalize, find every non-orthogal Ux to compute the seach
%directions dUx
V{d} = U{d};
for i = d:-1:2
    [Q, R] = qr(v2h(V{i}, m(i))', 'econ');
    V{i} = h2v(Q', m(i));
    V{i-1} = U{i-1} * R';
    Ux{i-1} = V{i-1};
end


[~,yl] = Ax_left(A,U,d);
[~,yr] = Ax_right(A,V,1);

dUx = cell(d,1);

%solve the seach directions
for i = 1:d
    dUx{i} = TTcore_Newton(yl{i},yr{i},A{i},Ux{i},residual,lambda);
end

%solve A*dx for computating step sizes
Adx = zeros(n_samples,d);
for i = 1:d
    dUi = reshape(dUx{i},[r(i), m(i), r(i+1)]);
    dUi = reshape(permute(dUi, [2 1 3]),m(i),[]);
    AdU = A{i}*dUi;
    AdU = reshape(AdU,n_samples,r(i),r(i+1));

    Adxi = zeros(n_samples,r(i+1));
    for j = 1:r(i+1)
        Adxi(:,j) = sum(yl{i}.*AdU(:,:,j),2);
    end
    Adx(:,i) = sum(Adxi.*yr{i},2);
end


if beta <= 0
    %no momentum

    %find Gram matrix with regularization
    reg_matrix = zeros(d,d);
    for i = 1:d-1
        reg_matrix(i,i) = norm(dUx{i},'fro')^2;
        UdU = dUx{i}'*U{i};
        for j =i+1:d-1
            reg_matrix(i,j) = sum(conj(V{j}) .* h2v(UdU * v2h(dUx{j}, m(j)), m(j)),'all');
            UdU = V{j}' * h2v(UdU * v2h(U{j}, m(j)), m(j));
        end
        reg_matrix(i,d) = sum(conj(V{d}) .* h2v(UdU * v2h(dUx{d}, m(d)), m(d)),'all');
    end
    reg_matrix(d,d) = norm(dUx{d},'fro')^2;
    reg_matrix = lambda^2*reg_matrix;
    reg_matrix = reg_matrix + triu(reg_matrix,1)';

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
    Adx_old = zeros(n_samples,d);

    for i = 1:d
        dUi = reshape(dU_old{i},[r(i), m(i), r(i+1)]);
        dUi = reshape(permute(dUi, [2 1 3]),m(i),[]);
        AdU = A{i}*dUi;
        AdU = reshape(AdU,n_samples,r(i),r(i+1));

        Adxi = zeros(n_samples,r(i+1));
        for j = 1:r(i+1)
            Adxi(:,j) = sum(yl{i}.*AdU(:,:,j),2);
        end
        Adx_old(:,i) = sum(Adxi.*yr{i},2);
    end
    Adx = [Adx Adx_old];

    %find Gram matrix with regularization
    reg_matrix = zeros(d,d);
    for i = 1:d-1
        reg_matrix(i,i) = norm(dUx{i},'fro')^2;
        reg_matrix(i+d,i+d) = norm(dU_old{i},'fro')^2;
        reg_matrix(i,i+d) = sum(conj(dUx{i}).*dU_old{i},"all");
        UdU1 = dUx{i}'*U{i};
        UdU2 = dU_old{i}'*U{i};
        for j =i+1:d-1
            reg_matrix(i,j) = sum(conj(V{j}) .* h2v(UdU1 * v2h(dUx{j}, m(j)), m(j)),'all');
            reg_matrix(i+d,j+d) = sum(conj(V{j}) .* h2v(UdU2 * v2h(dU_old{j}, m(j)), m(j)),'all');
            reg_matrix(i,j+d) = sum(conj(V{j}) .* h2v(UdU1 * v2h(dU_old{j}, m(j)), m(j)),'all');
            reg_matrix(j,i+d) = conj(sum(conj(V{j}) .* h2v(UdU2 * v2h(dUx{j}, m(j)), m(j)),'all'));

            UdU1 = V{j}' * h2v(UdU1 * v2h(U{j}, m(j)), m(j));
            UdU2 = V{j}' * h2v(UdU2 * v2h(U{j}, m(j)), m(j));
        end
        reg_matrix(i,d) = sum(conj(V{d}) .* h2v(UdU1 * v2h(dUx{d}, m(d)), m(d)),'all');
        reg_matrix(i+d,2*d) = sum(conj(V{d}) .* h2v(UdU2 * v2h(dU_old{d}, m(d)), m(d)),'all');
        reg_matrix(i,2*d) = sum(conj(V{d}) .* h2v(UdU1 * v2h(dU_old{d}, m(d)), m(d)),'all');
        reg_matrix(d,i+d) = conj(sum(conj(V{d}) .* h2v(UdU2 * v2h(dUx{d}, m(d)), m(d)),'all'));
    end
    reg_matrix(d,d) = norm(dUx{d},'fro')^2;
    reg_matrix(2*d,2*d) = norm(dU_old{d},'fro')^2;
    reg_matrix(d,2*d) = sum(conj(dUx{d}).*dU_old{d},"all");

    reg_matrix = lambda^2*reg_matrix;
    reg_matrix = reg_matrix + triu(reg_matrix,1)';

    reg_vec = zeros(d,1);
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

