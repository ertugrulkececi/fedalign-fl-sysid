clear;clc

addpath(fullfile(pwd,'libs'));
%% Hair Dryer Dataset

load dryer2

dry = iddata(y2,u2,0.1);

ze = dry(1:300);
ze = detrend(ze);
zv = dry(800:900);   % select an independent data set for validation
zv = detrend(zv);

sigma = 0.05;
nx = 3;

%% MR Damper Dataset

% load mrdamper.mat
% 
% data = iddata(F,V,0.1);
% ze = data(1:3000);
% zv = data(3001:end);
% 
% sigma = 5;
% nx = 3;

%% Training Settings

numRuns = 20;
numRounds = 20;
numWorkers = 20;

opt = ssestOptions;
opt.SearchOptions.MaxIterations = 1;

K = size(ze,1);

%% Training Loop

seeds = 1:numRuns;
for runs=1:numRuns
    rng(seeds(runs))
    for i=1:20 %Dataset for each worker
        u(:,i) = ze.InputData;
        y(:,i) = ze.OutputData + normrnd(0,sigma,[K,1]);
    end
    for rounds = 1:numRounds
        if rounds == 1
            for i = 1:numWorkers %Workers
                sys{rounds,i} = ssest(u(:,i),y(:,i),nx,"Ts",0.1,"DisturbanceModel","None",opt);
                lfit(runs,rounds,i) = sys{rounds,i}.Report.Fit.FitPercent;
                temp = zpk(sys{rounds,i});
                gain(runs,rounds,i) = dcgain(temp);
            end
            %Center Server
            [T,sysG] = FedAlignA(sys,rounds,numWorkers);
            sysGlog{runs,rounds,1} = sysG;
        else
            for i=1:numWorkers %Workers
                tempG = sysG{1,1};
                init_sys = idss(T{i}*tempG.A*inv(T{i}),T{i}*tempG.B,tempG.C*inv(T{i}),0,zeros(nx,1),[],0.1);
                sys{rounds,i} = ssest(u(:,i),y(:,i),init_sys,"DisturbanceModel","None",opt);
                lfit(runs,rounds,i) = sys{rounds,i}.Report.Fit.FitPercent;
            end
            %Center Server
            [T,sysG] = FedAlignA(sys,rounds,numWorkers);
            sysGlog{runs,rounds,1} = sysG;
        end
    end
    for i=1:numWorkers %Workers
        tempG = sysG{1,1};
        init_sys = idss(T{i}*tempG.A*inv(T{i}),T{i}*tempG.B,tempG.C*inv(T{i}),0,zeros(nx,1),[],0.1);
        [~,gfit(runs,i),~] = compare(u(:,i),y(:,i),init_sys);
        [~,gfitV(runs,i),~] = compare(zv,init_sys);
    end
end

%% Metrics 

meanBFR = mean(mean(gfit,2));
stdBFR = std(mean(gfit,2));

meanBFR_Val = mean(mean(gfitV,2));
stdBFR_Val = std(mean(gfitV,2));

fprintf("Training BFR: %.2f ± (%.2f)  |  Test BFR: %.2f ± (%.2f)\n", meanBFR, stdBFR, meanBFR_Val, stdBFR_Val);