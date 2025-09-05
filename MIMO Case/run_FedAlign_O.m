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

opt = ssestOptions;
opt.SearchOptions.MaxIterations = 1;

K = length(U);

%% Training Loop

for runs=1:numRuns
    clear sys lfit gfit 
    rng(runs)
    for i=1:numWorkers
        u{i} = U(1:K,:);
        y{i} = Y(1:K,:) + normrnd(0,sigma,[K,ny]);
    end
    for rounds = 1:numRounds
        if rounds == 1
            for i = 1:numWorkers %Workers
                sys{rounds,i} = ssest(u{i},y{i},nx,"Ts",0.1,"DisturbanceModel","None",opt);
                lfit(rounds,:,i) = sys{rounds,i}.Report.Fit.FitPercent;
            end
            [T,sysG,index] = FedAlignO(sys,rounds,runs,numWorkers,UVal);
            sysGlog{runs,rounds,1} = sysG;
        else
            for i=1:numWorkers %Workers
                tempG = sysG{1,1};
                init_sys = idss(T{i}*tempG.A*inv(T{i}),T{i}*tempG.B,tempG.C*inv(T{i}),zeros(ny,nu),zeros(nx,ny),[],0.1);
                sys{rounds,i} = ssest(u{i},y{i},init_sys,"DisturbanceModel","None",opt);
                lfit(rounds,:,i) = sys{rounds,i}.Report.Fit.FitPercent;
            end
            %Center Server
            [T,sysG,index] = FedAlignO(sys,rounds,runs,numWorkers,UVal);
            sysGlog{runs,rounds,1} = sysG;
        end

    end
    for i=1:numWorkers %Workers
        tempG = sysG{1,1};
        init_sys = idss(T{i}*tempG.A*inv(T{i}),T{i}*tempG.B,tempG.C*inv(T{i}),zeros(ny,nu),zeros(nx,ny),[],0.1);
        [~,gfit(:,i),~] = compare(u{i},y{i},init_sys);
        [~,gfitVal(:,i),~] = compare(UVal,YVal,init_sys);
    end
    lfit_log{runs} = lfit;
    gfit_log{runs} = gfit;
    gfitVal_log{runs} = gfitVal;
end

%% Metrics

for i=1:numRuns
    Gfit(i,:,:) = gfit_log{1,i};
    GfitVal(i,:,:) = gfitVal_log{1,i};
end

for j=1:ny
    meanBFR(j) = mean(mean(squeeze(Gfit(:,j,:)),2));
    stdBFR(j) = std(mean(squeeze(Gfit(:,j,:)),2));
    meanBFRVal(j) = mean(mean(squeeze(GfitVal(:,j,:)),2));
    stdBFRVal(j) = std(mean(squeeze(GfitVal(:,j,:)),2));
    fprintf("Output %d >> Training BFR: %.2f ± (%.2f)  |  Test BFR: %.2f ± (%.2f)\n", j, meanBFR(j), stdBFR(j), meanBFRVal(j), stdBFRVal(j));
end


