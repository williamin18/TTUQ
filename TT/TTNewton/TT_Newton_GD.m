function [x,training_err,test_err,epoch] = TT_Newton_GD(A,b,x,rank,tol,max_iterations,A_test,b_test,lambda)
%x: unknown vector in TT-format
%A: left hand side matrix, rows in rank-1 format
%b: right hand side vector
%lambda: regularization parameter, it is equal to sqrt(lambda) in the paper


%parameter initialization
d = length(A);
[n_samples,~] = size(A{1});
[~,m,~] = TTsizes(x);

x = TTorthogonalizeLR(x);
r = b - multi_r1_times_TT(A,x);
dx_TT = 0;
beta = 0;

break_counter = 0;
break_limit = 5;
err_old = 100;

for epoch = 1:max_iterations

    %Compute Newton updates for each core
    [V,dUx] = TT_Newton_Gradient(A,x,r,beta,dx_TT,lambda);
    
    %Update x by TT-structure update
    dx_TT = TT_Riemannian_fromGTensor(x,V,dUx);
    x = TT_Riemannian_update(x,V,dUx,1,rank);
    r = b - multi_r1_times_TT(A,x);
    beta = 1; %momentum starts after the first iteration

    
    
    
    training_err = norm(r)/norm(b);
    r_test = multi_r1_times_TT(A_test,x) - b_test;
    test_err = norm(r_test)/norm(b_test);

    if test_err < tol || test_err/training_err>4
        break
    end

    if   err_old-training_err < tol/1000*d
        break_counter = break_counter+1;
        if break_counter > break_limit
            break
        end
    else
        break_counter = 0;
    end
    err_old = training_err;
end
end

