function [x,y,training_err,test_err,epoch] = TT_Newton_rational_r1d3(A,x,C,y,b,rank,tol,max_iterations,A_test,C_test,b_test,lambda,batch_size)
%find x_n and x_d that minimizes || (A*x_1)./(1+A*x_2)-b||, x = [x1; x2]


%parameter initialization
break_counter = 0;
break_limit = 5;
err_old = 100;

x = TTorthogonalizeRL(x);
y = TTorthogonalizeRL(y);

x = TTorthogonalizeLR(x);
y = TTorthogonalizeLR(y);

d = length(A);
[n_samples,~] = size(A{1});
[~,m,~] = TTsizes(x);


[n_test_samples,~] = size(A_test{1});


beta = 0;
dx_TT = 0;
dy_TT = 0;
% 
% A1 = A;
% b1 = b;

best_test_err = 100;
best_x = 0;
best_y = 0;

if nargin == 12

    Ax = multi_r1_times_TT(A,x);
    Cy = multi_r1_times_TT(C,y);
    r = b - Ax./(1+Cy);

    for epoch = 1:max_iterations
    
        
        % rho = 1./(1+C*y);
        % A1{1} = A{1}.*rho;
        % b1 = b.*rho;
    
        
        %Compute Newton updates for each core
        [x,V,dU,Adx,y,W,dY,Cdy] = TT_Newton_Gradient_rational_r1d3(A,C,x,y,b,r,beta,dx_TT,dy_TT,lambda);
        %Update x by TT-structure update
        [dx_TT,x2] = TT_Riemannian_fromGTensor(x,V,dU);
        [dy_TT,y2] = TT_Riemannian_fromGTensor(y,W,dY);
        beta = 1;
    
        % x = TT_Riemannian_update(x,V,dUx,1,rank);
        
        % [x,y,Ax,Cy] = TT_rounding_ALS_RPC(A,x2,C,y2,b,rank,lambda);
        x = TTrounding(x2,1e-5,rank);
        y = TTrounding(y2,1e-5,rank);
        % 
        % [y, Cy] = TT_rounding_ALS(C,y2,(Ax+Adx)./b-1,rank,lambda);
        % [x, Ax] = TT_rounding_ALS(A,x2,b.*(1+Cy),rank,lambda);
    
    
        % r = b - Ax./(1+Cy);
        r = b  - multi_r1_times_TT(A,x)./(1+multi_r1_times_TT(C,y));
        r_test = b_test - multi_r1_times_TT(A_test,x)./(1+multi_r1_times_TT(C_test,y));
        training_err = norm(r)/norm(b);
        test_err = norm(r_test)/norm(b_test);
    
        % [ training_err test_err]
        if test_err<best_test_err
            best_test_err = test_err;
            best_x = x;
            best_y = y;
        end
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
    x = best_x;
    y = best_y;
    test_err = best_test_err;
elseif nargin == 13 %use batches 
    
    for epoch = 1:max_iterations
        shuffle_idx = randperm(n_samples);
        for i =1:d
            A{i} = A{i}(shuffle_idx,:);
            C{i} = C{i}(shuffle_idx,:);
        end
        b = b(shuffle_idx);
        %batch_split
        max_batches = floor(n_samples/batch_size);
        training_err = 0;
        for batch_idx = 1:max_batches
            batch_A= cell(d,1);
            batch_C = cell(d,1);
            batch_b = b((batch_idx-1)*batch_size+1:batch_idx*batch_size);
            for i = 1:d
                batch_A{i} = A{i}((batch_idx-1)*batch_size+1:batch_idx*batch_size, :);
                batch_C{i} = C{i}((batch_idx-1)*batch_size+1:batch_idx*batch_size, :);
            end

            % Compute the residual for the current batch
            r = batch_b - multi_r1_times_TT(batch_A, x)./(1 + multi_r1_times_TT(batch_C, y));
            % Continue with the Newton updates...

            % Compute the Newton updates for the current batch
            [x,V,dU,Adx,y,W,dY,Cdy] = TT_Newton_Gradient_rational_r1d3(batch_A, batch_C, x, y, batch_b, r, beta, dx_TT, dy_TT, lambda);
            
            %Update x by TT-structure update
            [dx_TT,x2] = TT_Riemannian_fromGTensor(x,V,dU);
            [dy_TT,y2] = TT_Riemannian_fromGTensor(y,W,dY);
            beta = 1;
            x = TTrounding(x2,1e-5,rank);
            y = TTrounding(y2,1e-5,rank);

            training_err = training_err + norm(r)^2;
        end
        training_err = sqrt(training_err)/norm(b);
        r_test = b_test - multi_r1_times_TT(A_test,x)./(1+multi_r1_times_TT(C_test,y));
        test_err = norm(r_test)/norm(b_test);

        if test_err < tol 
            break
        end
        
        [ training_err test_err]
        if test_err<best_test_err
            best_test_err = test_err;
            best_x = x;
            best_y = y;
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
    x = best_x;
    y = best_y;
    test_err = best_test_err;
end

end