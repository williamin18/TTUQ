function [Cy] = Cy_r1(C,y)
%CY_R1 Summary of this function goes here
%   Detailed explanation goes here
    [n_samples,d] = size(C);
    Cy = ones(n_samples,1);
    for i = 1:d
        Cy = Cy.*(1+C(:,i)*y(i));
    end
end

