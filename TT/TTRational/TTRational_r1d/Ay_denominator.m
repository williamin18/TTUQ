function [Ady] = Ay_denominator(A,y,r)
%AY_R1 undefined
%   undefined
    [n_samples,~] = size(A{1});
    d = length(y);

    yl = zeros(n_samples,d);
    yr = zeros(n_samples,d);

    yl(:,1) = ones(n_samples,1);
    for i = 2:d
        yl(:,i) = yl(:,i-1).*(A{i-1}(:,1) + A{i-1}(:,2)*y(i-1));
    end
    yr(:,d) = ones(n_samples,1);
    for i = d-1:-1:1
        yr(:,i) = (A{i+1}(:,1) + A{i+1}(:,2)*y(i+1)).*yr(:,i+1);
    end

    Ady = zeros(n_samples,d);
    for i = 1:d
        Ady(:,i) = yl(:,i).* A{i}(:,2).*yr(:,i);
    end
    
end