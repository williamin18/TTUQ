function [x,training_err,test_err,epoch] = TT_RPC_SGD(A,x,b,rank,tol,max_iterations,A_test,b_test,lambda)
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


[n_test_samples,~] = size(A_test{1});
A_test_linear = [A_test;{[ones(n_test_samples,1) -b_test]}];
for i = 1:d
    A_test_linear{i}(n_test_samples+1,:) = [1 zeros(1,m(i)-1)];
end
A_test_linear{d+1}(n_test_samples+1,:) = [0 1];
b_test_linear = [zeros(n_test_samples,1);1];


beta = 0;
dx_TT = 0;


batch_size = 20;
max_epoch = 10;
for epoch = 1:max_epoch

    new_order = randperm(n_samples);
    A2 = cell(d,1);
    for i = 1:d
        A2{i} = A{i}(new_order,:);
    end
    b2 = b(new_order);

    for j = 1:batch_size:n_samples-batch_size+1
        A_j = cell(d+1,1);
        for i = 1:d
            A_j{i} = A2{i}(j:j+batch_size-1,:);
            A_j{i}(batch_size+1,:) = [1 zeros(1,m(i)-1)];
        end
        b_j = b2(j:j+batch_size-1);
        A_j{d+1} = [ones(n_samples,1) -b_j; 0 1];
        bj_linear = [zeros(batch_size,1); 1];

        r_j = bj_linear - multi_r1_times_TT(A_j,x);
        df = multi_r1_times_vec_to_TT(A_j,r_j);
        Adf = multi_r1_times_TT(A2,df);

        r = b_linear - multi_r1_times_TT(A_linear,x);
        step_size = Adf'*r/(Adf'*Adf);

        x = TTaxby(1,x,step_size,df);
        x = TTrounding_Randomize_then_Orthogonalize(x,[1 r_round*ones(1,d-1) 1]);


        x_n = x(1:d);
        x_n{d} = x_n{d}*x{d+1}(1:tt_ranks(d+1));
        x_d = x(1:d);
        x_d{d} = x_d{d}*x{d+1}(tt_ranks(d+1)+1:2*tt_ranks(d+1));
        r_train = b - multi_r1_times_TT(A,x_n)./multi_r1_times_TT(A,x_d);
        r_test = b_test - multi_r1_times_TT(A_test,x_n)./multi_r1_times_TT(A_test,x_d);
        training_err = norm(r_train)/norm(b);
        test_err = norm(r_test)/norm(b_test);
    end
end

tt_ranks = TTranks(x);
x_n = x(1:d);
x_n{d} = x_n{d}*x{d+1}(1:tt_ranks(d+1));
x_d = x(1:d);
x_d{d} = x_d{d}*x{d+1}(tt_ranks(d+1)+1:2*tt_ranks(d+1));

r_train = b2 - multi_r1_times_TT(A2,x_n)./multi_r1_times_TT(A2,x_d);
r_test = b_test - multi_r1_times_TT(A_test,x_n)./multi_r1_times_TT(A_test,x_d);
training_err = norm(r_train)/norm(b2);
test_err = norm(r_test)/norm(b_test);

end

