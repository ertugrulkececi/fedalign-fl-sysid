clear;clc;

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

options = ssestOptions;
options.SearchOptions.MaxIterations = 1;

K = size(ze,1);

unstableflag = zeros(numRuns,1);

%% Training Loop

seeds = 1:numRuns;
for runs = 1:numRuns
    rng(seeds(runs))
    for i=1:20 %Dataset for each worker
        u(:,i) = ze.InputData;
        y(:,i) = ze.OutputData + normrnd(0,sigma,[K,1]);
    end
    for rounds = 1:numRounds

        if rounds == 1
            for i = 1:numWorkers %Workers
                sys{rounds,i} = ssest(u(:,i),y(:,i),nx,"Ts",0.1,"DisturbanceModel","None",options);
                lfit(runs,rounds,i) = sys{rounds,i}.Report.Fit.FitPercent;
                temp = zpk(sys{rounds,i});
            end
            %Center Server
            sysG = FedAvg(sys,rounds,numWorkers);
            sysGlog{runs,rounds,1} = sysG;

            if any(abs(eig(sysG))> 1)
                unstableflag(runs,:) = rounds;
                break
            end
        else
            for i=1:numWorkers %Workers
                init_sys = idss(sysG.A,sysG.B,sysG.C,0,zeros(nx,1),[],0.1);
                sys{rounds,i} = ssest(u(:,i),y(:,i),init_sys,"DisturbanceModel","None",options);
                lfit(runs,rounds,i) = sys{rounds,i}.Report.Fit.FitPercent;
                temp = zpk(sys{rounds,i});
            end
            %Center Server
            sysG = FedAvg(sys,rounds,numWorkers);
            sysGlog{runs,rounds,1} = sysG;

            if any(abs(eig(sysG))> 1)
                unstableflag(runs,:) = rounds;
                break
            end
        end

    end

    for i=1:numWorkers
        [~,gfit(runs,i),~] = compare(u(:,i),y(:,i),sysG);
    end
    [~,gfitV(runs),~] = compare(zv,sysG);
end

%% Metrics 

k=1;
uns = 0;
f2l = 0;
for i=1:numRuns
    if unstableflag(i) == 0 && mean(gfit(i,:),"all") > 0
        gfitS(k,:) = gfit(i,:);
        gfitValS(k) = gfitV(1,i);
        k=k+1;
    elseif unstableflag(i) ~= 0
        uns =  uns+1;
    else
        f2l = f2l+1;
    end
end

fprintf("Number of Unstable Global Models: %d | Number of Failed2Learn Global Models: %d\n", uns, f2l);

meanBFR = mean(mean(gfitS,2));
stdBFR = std(mean(gfitS,2));

meanBFR_Val = mean(gfitValS);
stdBFR_Val = std(gfitValS);

fprintf("Training BFR: %.2f ± (%.2f)  |  Test BFR: %.2f ± (%.2f)\n", meanBFR, stdBFR, meanBFR_Val, stdBFR_Val);

