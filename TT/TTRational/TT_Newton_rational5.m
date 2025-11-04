function [x,training_err,test_err,outer_iteration] = TT_Newton_rational5(A,x,b,rank,tol,max_iterations,A_test,b_test,lambda)
%find x_n and x_d that minimizes || (A*x_1)./(1+A*x_2)-b||, x = [x1; x2]


%parameter initialization
break_counter = 0;
break_limit = 3;
err_old = 100;


x = TTorthogonalizeLR(x);
d = length(A);
[n_samples,~] = size(A{1});
[n_test_samples,~] = size(A_test{1});
[~,m,~] = TTsizes(x);

A_linear = [A;{[ones(n_samples,1) -b]}];
for i = 1:d
    A_linear{i}(n_samples+1,:) = [1 zeros(1,m(i)-1)];
end
A_linear{d+1}(n_samples+1,:) = [0 1];
b_linear = [zeros(n_samples,1);1];

A_test_linear = [A_test;{[ones(n_test_samples,1) -b_test]}];
for i = 1:d
    A_test_linear{i}(n_test_samples+1,:) = [1 zeros(1,m(i)-1)];
end
A_test_linear{d+1}(n_test_samples+1,:) = [0 1];
b_test_linear = [zeros(n_test_samples,1);1];

beta = 0;
dx_TT = 0;

r = b_linear - multi_r1_times_TT(A_linear,x);

for outer_iteration = 1:10
    for inner_iteration = 1:10
        [~,~,tt_ranks] = TTsizes(x);
        %Compute Newton updates for each core
        [V,dUx] = TT_Newton_Gradient(A_linear,x,r,beta,dx_TT,lambda);

        %Update x by TT-structure update
        beta = 1;
        dx_TT = TT_Riemannian_fromGTensor(x,V,dUx);
        x = TT_Riemannian_update(x,V,dUx,1,rank);

        r = b_linear - multi_r1_times_TT(A_linear,x);
        training_err = norm(r)/norm(b_linear);
        r_test = multi_r1_times_TT(A_test_linear,x) - b_test_linear;
        test_err = norm(r_test)/norm(b_test_linear);



        if test_err < tol || test_err/training_err>4
            break
        end
        if   err_old-training_err < tol/100*d
            break_counter = break_counter+1;
            if break_counter > break_limit
                break
            end
        else
            break_counter = 0;
        end
    end


    
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

    rho = 1./(multi_r1_times_TT(A,x_d));
    rho = rho/max(abs(rho));
    A_linear{d+1}(1:n_samples,:) = [rho -rho.*b];

    rho_test = 1./(multi_r1_times_TT(A_test,x_d));
    rho_test = rho_test/max(abs(rho_test));
    A_test_linear{d+1}(1:n_test_samples,:) = [rho_test -rho_test.*b_test];
end


end


