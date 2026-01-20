function [x,y,training_err,test_err,epoch] = TT_Newton_rational6(A,x,b,y,rank,tol,max_iterations,A_test,b_test,lambda)
%find x_n and x_d that minimizes || (A*x_1)./(1+A*x_2)-b||, x = [x1; x2]


%parameter initialization
break_counter = 0;
break_limit = 5;
err_old = 100;


x = TTorthogonalizeLR(x);
d = length(A);
[n_samples,~] = size(A{1});
[~,m,~] = TTsizes(x);

C = zeros(n_samples,d);
for i = 1:d
    C(:,i) = A{i}(:,2);
end
Cb = C.*b;

[n_test_samples,~] = size(A_test{1});
C_test = zeros(n_test_samples,1);
for i = 1:d
    C_test(:,i) = A_test{i}(:,2);
end
beta = 0;
dx_TT = 0;

for epoch = 1:max_iterations

    


    r = b_linear - multi_r1_times_TT(A_linear,x) + Cb*y;
    test_r1 =  norm(r);
    
    %Compute Newton updates for each core
    [V,dUx,y] = TT_Newton_Gradient_rational(A,Cb,x,y,r,beta,dx_TT,lambda);
    %Update x by TT-structure update
    dx_TT = TT_Riemannian_fromGTensor(x,V,dUx);
    beta = 1;
    x = TT_Riemannian_update(x,V,dUx,1,rank);

    % [test_r1 norm(b_linear - multi_r1_times_TT(A_linear,x))]

    r_train = b - multi_r1_times_TT(A,x)./(C*y);
    r_test = b_test - multi_r1_times_TT(A_test,x)./(C_test*y);
    training_err = norm(r_train)/norm(b);
    test_err = norm(r_test)/norm(b_test);

    [test_r1 training_err test_err];
    if test_err < tol 
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

