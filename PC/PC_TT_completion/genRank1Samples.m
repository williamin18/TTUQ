function [sample_xi] = genRank1Samples(m,d,polynomial)
%GENRANK1SAMPLES Summary of this function goes here
%   Detailed explanation goes here
switch polynomial
    case "Hermite"
        f = @genHermite;
    otherwise
        err('Unsupported polynomial type')
end

xi_roots = genPolynomialRoots(m+1,polynomial);



% ref_points = round(m+1)/2*ones(1,d );
% 
% 
% sample_idx = zeros(d*m+1,d);
% sample_idx(1,:) = ref_points;
% counter = 2;
% for i = 1:d
%     temp = ref_points;
%     for j = 1:m+1
%         if j~= ref_points(i)
%             temp(i) = j;
%             sample_idx(counter,:) = temp;
%             counter = counter+1;
%         end
%     end
% end
% sample_xi = xi_roots(sample_idx);


sample_xi = zeros(d*(m+1),d);
for i = 1:d
    sample_xi((i-1)*(m+1)+1:i*(m+1),i) = xi_roots;
end
end

