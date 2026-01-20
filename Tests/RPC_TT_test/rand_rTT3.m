n = 4;
d = 10;
N = n*ones(d,1);
r = 3;

n_samples = 4000;
n_test_samples = 200;

X_true = TTrand([N ;2],r);

X_true = TTorthogonalizeRL(X_true);
X_true = TTorthogonalizeLR(X_true);
X_true{d+1} = X_true{d+1}/norm( X_true{d+1},'fro')*100;

[~,N,r] = TTsizes(X_true);
d1 = TT_eval(X_true,[ones(1,d) 2]);
X_true{d+1}= X_true{d+1}/d1;


Xn_true = X_true(1:d);
Xn_true{d} = Xn_true{d}*X_true{d+1}(1:r(d+1));

Xd_true = X_true(1:d);
Xd_true{d} = Xd_true{d}*X_true{d+1}(r(d+1)+1:2*r(d+1));


xi = randn(n_samples,d);
xi2 = randn(n_test_samples,d);


H = genPolynomialSamplesTensor(xi,n-1,'Hermite');
H_test = genPolynomialSamplesTensor(xi2,n-1,'Hermite');

lambda1 = 0.3;
for i = 1:d
    for j = 1:n-1
        H{i}(:,j+1) = H{i}(:,j+1)*lambda1^j;
        H_test{i}(:,j+1) = H_test{i}(:,j+1)*lambda1^j;
    end
end

c = multi_r1_times_TT(H,Xn_true)./multi_r1_times_TT(H,Xd_true)+0.01*randn(n_samples,1);
c_test = multi_r1_times_TT(H_test,Xn_true)./multi_r1_times_TT(H_test,Xd_true);


x = TTrand(N,r);
x = TTorthogonalizeRL(x);
x = TTorthogonalizeLR(x);


% [y_predict,n_rpc,d_rpc,n_ierations] = rpc_total(xi,c,xi2,3,'Hermite');
% norm(y_predict-c_test)/norm(c_test)


[x,training_err,test_err,epoch] = TT_Newton_rational4(H,x,c,4,1e-4,100,H_test,c_test,0.1);
[training_err,test_err,epoch]

