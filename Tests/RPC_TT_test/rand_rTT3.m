n = 4;
d = 10;
N = n*ones(d,1);
r = 3;

n_samples = 2000;
n_test_samples = 200;

X_true = TTrand(N,r);

X_true = TTorthogonalizeRL(X_true);
X_true = TTorthogonalizeLR(X_true);
X_true{d} = X_true{d}/norm( X_true{d},'fro');

y_true = 0.3*randn(d,1)

A = cell(d,1);
A_test = cell(d,1);
C = zeros(n_samples,d);
C_test = zeros(n_test_samples,d);
for i =1: d
    A{i} = [ones(n_samples,1) randn(n_samples,n-1)];
    A_test{i} = [ones(n_test_samples,1) rand(n_test_samples,n-1)];

    C(:,i) = A{i}(:,2);
    C_test(:,i) = A_test{i}(:,2);
end


b = multi_r1_times_TT(A,X_true)./(1+C*y_true);
b_test = multi_r1_times_TT(A_test,X_true)./(1+C_test*y_true);


x = TTrand(N,r);
x = TTorthogonalizeRL(x);
x = TTorthogonalizeLR(x);
y = zeros(d,1);


% [y_predict,n_rpc,d_rpc,n_ierations] = rpc_total(xi,c,xi2,3,'Hermite');
% norm(y_predict-c_test)/norm(c_test)

[x,y,training_err,test_err,epoch] = TT_Newton_rational6(A,x,b,y,r,1e-4,2000,A_test,b_test,0)
[training_err,test_err,epoch]

