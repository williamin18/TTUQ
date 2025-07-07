function [y_out,PC_coefficients] = pc_collocation_total(x_train,y_train,x_out,order,polynomial)
%SOLVECOEFFICIENTSTOTAL Summary of this function goes here
%   Detailed explanation goes here
switch polynomial
    case "Hermite"
        f = @genHermite;
    case "Legendre"
        f = @genLegendre;
    otherwise
        err('Unsupported polynomial type')
end


[n_train,d] = size(x_train);
H = genHmat_total(order,d);
[n_total,~] = size(H);
sample_polynomial_mat = ones(n_train,n_total);
for i = 1:n_train
    for j = 1:n_total
        for k = 1:d
            if H(j,k)>0
                sample_polynomial_mat(i,j) = sample_polynomial_mat(i,j)*f(x_train(i,k),H(j,k));
            end
        end
    end
end

PC_coefficients = sample_polynomial_mat\y_train;

[~,n_y] = size(y_train);
[n_samples,~] = size(x_out);
y_out = zeros(n_samples,n_y);
for i = 1:n_samples
    ksi = x_out(i,:);
    h = ones(n_total,1);

    for j = 1:n_total
        for k = 1:d
            if H(j,k)>0
                h(j)=h(j)*f(ksi(k),H(j,k));
            end
        end
    end
    y_out(i,:) = h'*PC_coefficients;
end
end

