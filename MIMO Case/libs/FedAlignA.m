function [T,invT,sysG] = FedAlignA(sys,rounds,numWorkers)

%Num states,inputs,outputs
[nx,nu] = size(sys{rounds,1}.B);
[ny,~] = size(sys{rounds,1}.C);

globalA = zeros(nx,nx); globalB = zeros(nx,nu); globalC = zeros(ny,nx);
for i=1:numWorkers
   
    A = sys{rounds,i}.A;
    B = sys{rounds,i}.B;
    phi = ctrb(A,B(:,1));
    [U,S,V] = svd(phi);
    P1 = V*inv(S)*U';
    m = P1(nx,:);

    temp = zeros(nx, nx);
    temp(1,:) = m;    
    current_power = eye(nx); 
    for j = 2:nx
        current_power = current_power * A;  
        temp(j,:) = m * current_power;
    end
    [U,S,V] = svd(temp);
    T{i} = V*inv(S)*U';
    
    invT{i} = temp;

    globalA = globalA + invT{i} * (sys{rounds,i}.A *T{i});
    globalB = globalB + invT{i}*sys{rounds,i}.B;
    globalC = globalC + sys{rounds,i}.C * T{i};

end
globalA = globalA / numWorkers;
globalB = globalB / numWorkers;
globalC = globalC / numWorkers;

sysG{1,1} = ss(globalA,globalB,globalC,0,0.1);

end

