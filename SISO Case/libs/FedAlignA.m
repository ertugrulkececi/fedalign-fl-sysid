function [T,sysG,basin_index] = FedAlignA(sys,rounds,numWorkers)

%Num states,inputs,outputs
[nx,nu] = size(sys{rounds,1}.B);
[ny,~] = size(sys{rounds,1}.C);

T = cell(numWorkers,1);

syms s

globalA = zeros(nx,nx); globalB = zeros(nx,nu); globalC = zeros(ny,nx);
for i=1:20
    a = double(coeffs(det(s*eye(nx)-sys{rounds,i}.A)));
    P = ctrb(sys{rounds,i});
    T{i} = P*[a(2) a(3) 1; a(3) 1 0; 1 0 0]; % Right side of multiplication should be constructed according to nx
    globalA = globalA + inv(T{i})*sys{rounds,i}.A*T{i};
    globalB = globalB + inv(T{i})*sys{rounds,i}.B;
    globalC = globalC + sys{rounds,i}.C*T{i};
end
globalA = globalA / numWorkers;
globalB = globalB / numWorkers;
globalC = globalC / numWorkers;

sysG{1,1} = ss(globalA,globalB,globalC,0,0.1);




