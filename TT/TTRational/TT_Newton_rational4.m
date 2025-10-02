function [x,training_err,test_err,epoch] = TT_Newton_rational4(A,x,b,rank,tol,max_iterations,A_test,b_test,lambda)
%find x_n and x_d that minimizes || (A*x_1)./(1+A*x_2)-b||, x = [x1; x2]


%parameter initialization
break_counter = 0;
break_limit = 5;
err_old = 100;


x = TTorthogonalizeLR(x);
d = length(A);
[n_samples,~] = size(A{1});
[~,m,~] = TTsizes(x);

A_linear = [A;{[ones(n_samples,1) -b]}];
for i = 1:d
    A_linear{i}(n_samples+1,:) = [1 zeros(1,m(i)-1)];
end
A_linear{d+1}(n_samples+1,:) = [0 1];
b_linear = [zeros(n_samples,1);1];


beta = 0;
dx_TT = 0;

for epoch = 1:max_iterations

    

    [~,~,tt_ranks] = TTsizes(x);
    % x_d = x(1:d);
    % x_d{d} = x_d{d}*x{d+1}(tt_ranks(d+1)+1:2*tt_ranks(d+1));
    % rho = 1./(multi_r1_times_TT(A,x_d));
    % rho = rho/max(abs(rho));
    % A_linear{d+1}(1:n_samples,:) = [rho -rho.*b];
    r = b_linear - multi_r1_times_TT(A_linear,x);
    norm(r)
    
    %Compute Newton updates for each core
    [V,dUx] = TT_Newton_Gradient(A_linear,x,r,\beta,dx_TT,lambda);
    %Update x by TT-structure update
    dx_TT = TT_Riemannian_fromGTensor(x,V,dUx);
    beta = 1;
    x = TT_Riemannian_update(x,V,dUx,1,rank);

    
    x_n = x(1:d);
    x_n{d} = x_n{d}*x{d+1}(1:tt_ranks(d+1));
    x_d = x(1:d);
    x_d{d} = x_d{d}*x{d+1}(tt_ranks(d+1)+1:2*tt_ranks(d+1));
    r_train = b - multi_r1_times_TT(A,x_n)./multi_r1_times_TT(A,x_d);
    r_test = b_test - multi_r1_times_TT(A_test,x_n)./multi_r1_times_TT(A_test,x_d);
    training_err = norm(r_train)/norm(b);
    test_err = norm(r_test)/norm(b_test);

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

