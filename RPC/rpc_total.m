function [y_out,n_rpc,d_rpc,n_iterations] = rpc_total(x_train,y_train,x_out,order,polynomial,preprocessing_parameter,regularization_parameter,tol)
switch polynomial
    case "Hermite"
        f = @genHermite;
    case "Legendre"
        f = @genLegendre;
    otherwise
        err('Unsupported polynomial type')
end
lambda1 = preprocessing_parameter;

[n_train,d] = size(x_train);
H = genHmat_total(order,d);
[n_total,~] = size(H);
[~,n_y] = size(y_train);

sample_polynomial_mat = ones(n_train,n_total);
for i = 1:n_train
    for j = 1:n_total
        for k = 1:d
            if H(j,k)>0
                sample_polynomial_mat(i,j) = sample_polynomial_mat(i,j)*f(x_train(i,k),H(j,k))*lambda1^H(j,k);
            end
        end
    end
end
Phi = sample_polynomial_mat;

n_rpc = zeros(n_total,n_y);
d_rpc = zeros(n_total,n_y);
n_iterations = zeros(n_y,1);
for i = 1:n_y
    [n_rpc(:,i),d_rpc(:,i),n_iterations(i)] = sk_solve(Phi,y_train(:,i),tol,100,regularization_parameter);
end



[n_samples,~] = size(x_out);
y_out = zeros(n_samples,n_y);
for i = 1:n_samples
    ksi = x_out(i,:);
    h = ones(n_total,1);

    for j = 1:n_total
        for k = 1:d
            if H(j,k)>0
                h(j)=h(j)*f(ksi(k),H(j,k))*lambda1^H(j,k);
            end
        end
    end
    y_out(i,:) = (h'*n_rpc)./(h'*d_rpc);
end
end