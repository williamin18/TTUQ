function [G] = TTRegMatrix(dU,U,V,lambda,dU2)

%build the regulzation part of the Gram matrix
[d,m,~] = TTsizes(U);
G = zeros(d,d);
if nargin == 4
    for i = 1:d-1
        G(i,i) = sum(conj(dU{i}).*dU{i},"all");
        UdU = dU{i}'*U{i};
        for j =i+1:d-1
            G(i,j) = sum(conj(V{j}) .* h2v(UdU * v2h(dU{j}, m(j)), m(j)),'all');
            UdU = V{j}' * h2v(UdU * v2h(U{j}, m(j)), m(j));
        end
        G(i,d) = sum(conj(V{d}) .* h2v(UdU * v2h(dU{d}, m(d)), m(d)),'all');
    end
    G(d,d) = norm(dU{d},'fro')^2;
    G = lambda^2*G;
    G = G + triu(G,1)';
elseif nargin == 5
    for i = 1:d-1
        G(i,i) = norm(dU{i},'fro')^2;
        G(i+d,i+d) = norm(dU2{i},'fro')^2;
        G(i,i+d) = sum(conj(dU{i}).*dU2{i},"all");
        UdU1 = dU{i}'*U{i};
        UdU2 = dU2{i}'*U{i};
        for j =i+1:d-1
            G(i,j) = sum(conj(V{j}) .* h2v(UdU1 * v2h(dU{j}, m(j)), m(j)),'all');
            G(i+d,j+d) = sum(conj(V{j}) .* h2v(UdU2 * v2h(dU2{j}, m(j)), m(j)),'all');
            G(i,j+d) = sum(conj(V{j}) .* h2v(UdU1 * v2h(dU2{j}, m(j)), m(j)),'all');
            G(j,i+d) = conj(sum(conj(V{j}) .* h2v(UdU2 * v2h(dU{j}, m(j)), m(j)),'all'));

            UdU1 = V{j}' * h2v(UdU1 * v2h(U{j}, m(j)), m(j));
            UdU2 = V{j}' * h2v(UdU2 * v2h(U{j}, m(j)), m(j));
        end
        G(i,d) = sum(conj(V{d}) .* h2v(UdU1 * v2h(dU{d}, m(d)), m(d)),'all');
        G(i+d,2*d) = sum(conj(V{d}) .* h2v(UdU2 * v2h(dU2{d}, m(d)), m(d)),'all');
        G(i,2*d) = sum(conj(V{d}) .* h2v(UdU1 * v2h(dU2{d}, m(d)), m(d)),'all');
        G(d,i+d) = conj(sum(conj(V{d}) .* h2v(UdU2 * v2h(dU{d}, m(d)), m(d)),'all'));
    end
    G(d,d) = norm(dU{d},'fro')^2;
    G(2*d,2*d) = norm(dU2{d},'fro')^2;
    G(d,2*d) = sum(conj(dU{d}).*dU2{d},"all");

    G = lambda^2*G;
    G = G + triu(G,1)';
else
    error('unsupported number of inputs')
end



end

