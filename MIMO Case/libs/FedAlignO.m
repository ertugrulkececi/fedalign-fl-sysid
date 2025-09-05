function [T,sysG,index] = FedAlignO(sys,rounds,runs,numWorkers,Uval)

%Num states,inputs,outputs
[nx,nu] = size(sys{rounds,1}.B);
[ny,~] = size(sys{rounds,1}.C);

rng(runs)

index = randi([1 numWorkers]);
T = cell(numWorkers,1);
basin_index = zeros(numWorkers,1);
basin_index(index) = 1; 

t= 0:0.1:size(Uval,1)*0.1-0.1;
u = Uval';

y = cell(1,numWorkers);
x = cell(1,numWorkers);
for i = 1:numWorkers %Data üretimi
    [y{i}, t, x{i}] = lsim(sys{rounds,i},u,t');
    xtilde{i} = sys{rounds,i}.A*x{i}' + sys{rounds,i}.B*u;
end

for i=1:numWorkers
    if basin_index(i) == 0
        That = optimvar('That', nx, nx);

        objective = sum(sum((xtilde{i} - That*xtilde{index}).^2));

        prob = optimproblem;

        prob.Objective = objective;

        T0.That = zeros(nx, nx);

        options = optimoptions('lsqlin', 'Display', 'off');

        sol = solve(prob, T0, 'Options', options);

        T{i} = sol.That;
    else
        T{i} = eye(nx);
    end
end

globalA = zeros(nx,nx); globalB = zeros(nx,nu); globalC = zeros(ny,nx);
for i=1:numWorkers
    globalA = globalA + inv(T{i})*sys{rounds,i}.A*T{i};
    globalB = globalB + inv(T{i})*sys{rounds,i}.B;
    globalC = globalC + sys{rounds,i}.C*T{i};
end
globalA = globalA / numWorkers;
globalB = globalB / numWorkers;
globalC = globalC / numWorkers;
sysG{1,1} = ss(globalA,globalB,globalC,0,0.1);
end

