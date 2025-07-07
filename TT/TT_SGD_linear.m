function [x_min,training_err,test_err,n_iterations] = TT_SGD_linear(A,b,x,r_round,tol,max_epoches,A_test,b_test,lambda,batch_size)
%stupid method, implementd when I did not understand tensor train well, not
%suggested
d = length(A);
[n_samples,~] = size(A{1});
[n_test_samples,~] = size(A_test{1});
[~,m,~] = TTsizes(x);


break_counter = 0;
break_limit = 20;
err_min = norm(multi_r1_times_TT(A_test,x) - b_test)/norm(b_test);
% err_min = 100;
n_iterations = 0;
x_min = x;

for epoch = 1:max_epoches
    new_order = randperm(n_samples);
    for i = 1:d
        A{i} = A{i}(new_order,:);
    end
    b = b(new_order);
    for j = 1:batch_size:n_samples-batch_size+1
        A_j = cell(d,1);
        for i = 1:d
            A_j{i} = A{i}(j:j+batch_size-1,:);
        end
        b_j = b(j:j+batch_size-1);
        
        preconditioner = ones(batch_size,1);
        for i = 1:d
            preconditioner = preconditioner./reshape(dot(A_j{i},A_j{i},2),batch_size,1);
        end
        preconditioner = preconditioner.^0.5;
        preconditioner = diag(preconditioner);
        A_j{1} = preconditioner*A_j{1};
        b_j = preconditioner*b_j;


        
        r =  b_j - multi_r1_times_TT(A_j,x);
        training_err = norm(r)/norm(b);
        df = multi_r1_times_vec_to_TT(A_j,r);
        step_size = r'*r/TTdot(df,df);
        x = TTaxby(1,x,step_size,df);
        x = TTrounding_Randomize_then_Orthogonalize(x,[1 r_round*ones(1,d-1) 1]);
        

        n_iterations = n_iterations+1;

   

        r_test = multi_r1_times_TT(A_test,x) - b_test;
        test_err = norm(r_test)/norm(b_test);
        if test_err> err_min 
            break_counter = break_counter+1;
        else
            err_min = test_err;
            x_min = x;
            break_counter = 0;
        end

        if test_err < tol || break_counter > break_limit
            err_min = test_err;
            break
        end
    end

    r_test = multi_r1_times_TT(A_test,x) - b_test;
    test_err = norm(r_test)/norm(b_test);
    if test_err < tol || break_counter > break_limit
        break
    end
end
end

