function [sysG] = FedAvg(sys,rounds,numWorkers)

%Num states,inputs,outputs
[nx,nu] = size(sys{rounds,1}.B);
[ny,~] = size(sys{rounds,1}.C);

globalA = zeros(nx,nx); globalB = zeros(nx,nu); globalC = zeros(ny,nx);

for i = 1:numWorkers
    globalA = globalA + sys{rounds,i}.A;
    globalB = globalB + sys{rounds,i}.B;
    globalC = globalC + sys{rounds,i}.C;
end
globalA = globalA / numWorkers;
globalB = globalB / numWorkers;
globalC = globalC / numWorkers;

sysG = ss(globalA,globalB,globalC,0,0.1);

