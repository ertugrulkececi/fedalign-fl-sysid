clear;clc;

addpath(fullfile(pwd,'datasets'));
addpath(fullfile(pwd,'libs'));
%% Evaporator Dataset: 3 inputs - 3 outputs

% load evaporator.dat
% U = evaporator(:,1:3);
% Y = evaporator(:,4:6);
% UVal=U(3001:6000,:);
% YVal=Y(3001:6000,:);
% 
% U=U(1:3000,:);
% Y=Y(1:3000,:);
% 
% nx = 4;
% nu = size(U,2);
% ny = size(Y,2);
% 
% sigma = 0.1;

%% CD Player Arm Dataset: 2 inputs - 2 outputs

% load CD_player_arm.dat
% U = CD_player_arm(:,1:2);
% Y = CD_player_arm(:,3:4);
% U = normalize(U);
% Y = normalize(Y);
% 
% UVal = U(1201:end,:);
% YVal = Y(1201:end,:);
% 
% U = U(1:1200,:);
% Y = Y(1:1200,:);
% 
% nx = 2;
% nu = size(U,2);
% ny = size(Y,2);
% 
% sigma = 0.05;

%% Steam Eng Dataset: 2 inputs - 2 outputs

load steamEng
steam = iddata([GenVolt,Speed],[Pressure,MagVolt],0.05);
steam.InputName  = {'Pressure';'MagVolt'};
steam.OutputName = {'GenVolt';'Speed'};
U = steam.InputData;
Y = steam.OutputData;
UVal = U(251:end,:);
YVal = Y(251:end,:);
U = U(1:250,:);
Y = Y(1:250,:);

nx = 4;
nu = size(U,2);
ny = size(Y,2);

sigma = 0.001;

%% Training Settings

numRounds = 20;
numWorkers = 20;
numRuns = 20;

options = ssestOptions;
options.SearchOptions.MaxIterations = 1;

K = length(U);

unstableflag = zeros(numWorkers,1);

%% Training Loop

for runs = 1:numRuns
    rng(runs)
    clear sys lfit gfit gfitVal
    for i=1:numWorkers
        u{i} = U(1:K,:);
        y{i} = Y(1:K,:) + normrnd(0,sigma,[K,ny]);
    end
    for rounds = 1:numRounds
    
        if rounds == 1
            for i = 1:numWorkers %Workers
                sys{rounds,i} = ssest(u{i},y{i},nx,"Ts",0.1,"DisturbanceModel","None",options);
                lfit(rounds,:,i) = sys{rounds,i}.Report.Fit.FitPercent;
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
                tempG = sysG;
                init_sys = idss(tempG.A,tempG.B,tempG.C,zeros(ny,nu),zeros(nx,ny),[],0.1);
                sys{rounds,i} = ssest(u{i},y{i},init_sys,"DisturbanceModel","None",options);
                lfit(rounds,:,i) = sys{rounds,i}.Report.Fit.FitPercent;
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
        if any(abs(eig(sysG))> 1)
            gfit = 0;
            gfitVal= 0;
            break
        end
        [~,gfit(:,i),~] = compare(u{i},y{i},sysG);
        [~,gfitVal(:,i),~] = compare(UVal,YVal,sysG);
    end
    lfit_log{runs} = lfit;
    gfit_log{runs} = gfit;
    gfitVal_log{runs} = gfitVal;
end

%% Metrics

k=1;
uns = 0;
f2l = 0;
for i=1:numRuns
    if unstableflag(i) == 0 && mean(gfit_log{1,i},"all") > 0
        Gfit(k,:,:) = gfit_log{1,i};
        GfitVal(k,:,:) = gfitVal_log{1,i};
        k=k+1;
    elseif unstableflag(i) ~= 0
        uns =  uns+1;
    else
        f2l = f2l+1;
    end
end

fprintf("Number of Unstable Global Models: %d | Number of Failed2Learn Global Models: %d\n", uns, f2l);

for j=1:ny
    meanBFR(j) = mean(mean(squeeze(Gfit(:,j,:)),2));
    stdBFR(j) = std(mean(squeeze(Gfit(:,j,:)),2));
    meanBFRVal(j) = mean(mean(squeeze(GfitVal(:,j,:)),2));
    stdBFRVal(j) = std(mean(squeeze(GfitVal(:,j,:)),2));
    fprintf("Output %d >> Training BFR: %.2f ± (%.2f)  |  Test BFR: %.2f ± (%.2f)\n", j, meanBFR(j), stdBFR(j), meanBFRVal(j), stdBFRVal(j));
end