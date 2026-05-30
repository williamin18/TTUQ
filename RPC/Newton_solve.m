function [a_n,a_d,n_iterations,rel_err] = Newton_solve(Phi,y,a_n,a_d,tol,max_iterations,regularizaion_parameter)

rel_err_old = 1;
break_counter = 0;
break_limit = 5;

[n_samples,n_n] = size(Phi);
lambda = regularizaion_parameter;

Phi_n = Phi*a_n;
Phi_d = 1+Phi*a_d;
r = y - Phi_n./Phi_d;
for i = 1:max_iterations
    
    Jn = Phi./Phi_d;
    Jd = (-Phi_n./(Phi_d.^2)).*Phi;
    dx = [Jn Jd; lambda*eye(n_n*2)]\[r;-lambda*a_n;-lambda*a_d];
    % dx = [Jn Jd]\r;

    a_n = a_n + dx(1:n_n,:);
    a_d = a_d + dx(n_n+1:end,:);
    
    Phi_n = Phi*a_n;
    Phi_d = 1+Phi*a_d;
    r = y - Phi_n./Phi_d;

    rel_err = norm(r)/norm(y);
    
    
    if rel_err<tol 
        break
    end

    if  rel_err_old-rel_err < tol/10
        break_counter = break_counter+1;
        if break_counter >break_limit
            break
        end
    else
        break_counter = 0;
    end
    rel_err_old = rel_err;

end
n_iterations = i;
a_d(1) = a_d(1)+1;
end

