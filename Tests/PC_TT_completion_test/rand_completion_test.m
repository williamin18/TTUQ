n = 4;
d = 10;
N = n*ones(d,1);
r = 3;
n_samples = 2000;
n_test_samples = 100;



x_true = TTrand(N,3);
x_true = TTorthogonalizeLR(x_true);
x_true{d} = x_true{d}/norm( x_true{d},'fro');

A = randi(n,n_samples,d);
A = index2selector(A,n);
A_test = randi(n,n_test_samples,d);
A_test = index2selector(A_test,n);


x = TTrand(N,3);
x = TTorthogonalizeLR(x);
x{d} = x{d}/norm( x{d},'fro');



x_sample = multi_r1_times_TT(A,x_true);
x_test_sample =  multi_r1_times_TT(A_test,x_true);



[x,training_err,test_err,epoch] = TT_Riemannian_completion(A,x_sample,x,r,1e-4,2000,A_test,x_test_sample);
disp(epoch)
disp([training_err,test_err])
disp(TTnorm(TTaxby(1,x,-1,x_true))/TTnorm(x_true))