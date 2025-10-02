m = 3;
d = 8;
lambda1 = 0.3;
n_train = 5000;

H = genHmat_total(m,d);
f = @genHermite;
x_train = randn(n_train,d);

[n_total,~] = size(H);
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



A_train = sample_polynomial_mat(1:4000,:);
A_train2 = sample_polynomial_mat(1:4000,2:end);

A_test = sample_polynomial_mat(4001:end,:);
A_test2 = sample_polynomial_mat(4001:end,2:end);

n_true = rand(165,1);
d_true = 10*rand(164,1);

b = (A_train*n_true)./(1+A_train2*d_true)+0.01*rand(4000,1);
b_test = (A_test*n_true)./(1+A_test2*d_true);

[a_n1,a_d1,n_iterations1,rel_err1] = sk_solve(A_train,A_train2,b,1e-4,100);
[norm((A_test*a_n1)./(1+A_test2*a_d1) - b_test)/norm(b_test) n_iterations1 rel_err1]

[a_n2,a_d2,n_iterations2,rel_err2] = sk_solve2(A_train,b,1e-4,100);
[norm((A_test*a_n2)./(A_test*a_d2) - b_test)/norm(b_test) n_iterations2 rel_err2]


