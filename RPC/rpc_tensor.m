function [y_out,n_rpc,d_rpc,n_iterations] = rpc_tensor(x_train,y_train,x_out,polynomial)
switch polynomial
    case "Hermite"
        f = @genHermite;
    case "Legendre"
        f = @genLegendre;
    otherwise
        err('Unsupported polynomial type')
end


[n_train,d] = size(x_train);
[~,n_y] = size(y_train);

sample_polynomial_mat = ones(n_train,1);
n_tensor = 2^d;
for j = 1:d
    sample_polynomial_mat = [sample_polynomial_mat sample_polynomial_mat.*x_train(:,j)];
end
Phi = sample_polynomial_mat;

n_rpc = zeros(n_tensor,n_y);
d_rpc = zeros(n_tensor,n_y);
n_iterations = zeros(n_y,1);
for i = 1:n_y
    [n_rpc(:,i),d_rpc(:,i),n_iterations(i)] = sk_solve(Phi,y_train(:,i),5e-3,20,0);
end


[n_test,~] = size(x_out);
xi_test = ones(n_test,1);
for j = 1:d
    xi_test = [xi_test xi_test.*x_out(:,j)];
end
y_out = (xi_test*n_rpc)./(xi_test*d_rpc);
