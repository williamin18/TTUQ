function [y] = RPC_denominator_Riemannian_ALS_update(Ax,C,y,dy,b)
%UNTITLED undefined
%   undefined
[~,d]= size(C);

b = Ax./b;
Cy = cell(d,1);
Cy{d} = [1+C(:,d)*y(d), C(:,d)*dy(d)];

for i = d-1:-1:2
    Cy{i} = [Cy{i+1}(:,1).*(1+C(:,i)*y(i)), Cy{i+1}(:,1).*(C(:,i)*dy(i))+Cy{i+1}(:,2).*(1+C(:,i)*y(i))];
end

y(1) = ( b./(Cy{2}(:,1)+Cy{2}(:,2)) - 1 )./C(:,1);
Cy{1} = 1+C(:,1)*y(1);
for i = 2:d
    y(i) = ( b./(Cy{i+1}(:,1)+Cy{i+1}(:,2))./Cy{i-1} - 1 )./C(:,i);
    Cy{i} = Cy{i-1}.*(1+C(:,i)*y(i));
end

end