function [xFull,costs,activeBeams,activeNorms,topN,dimensionReductionTime] = BOO_IMRT_profile(A,ATrans,D,Weights,params)
    % This version attempts to increase t less often.
    
    % minimize sum_{b=1}^{numBeams} beamWeights(b) * || x_b ||_2
    % .5*mu || (A_0 x - minDose_target)_- ||_2^2
    % + sum_{i=0}^numOars .5*alphai || (A_i x - d_vecs{i})_+ ||_2^2
    % + sum_{i=1}^numOars .5*betai || A_i x ||^2 + eta || Dx ||_1^gamma
    % subject to x >= 0.
    
    % activeBeams contains the NEW indices of the active beams.
    % So you could never have 1162 be an active beam, because 1162 is not one
    % of the NEW indices.
    % (The new indices range from 1 to numValidBeams.  Of the 1162 beams, only
    % some are valid in the sense of not causing collisions.
    
    global root3 root6
    root3 = sqrt(3);
    root6 = sqrt(6);
    
    pruneTrigger = 40;
    % pruneTrigger = 10000; disp('WARNING!\nWARNING!\nWARNING! PRUNING IS OFF')
    t = params.stepSize;
    maxIter = params.maxIter;
    showTrigger = params.showTrigger;
    gamma = params.gamma; % Huber parameter
    eta = params.eta;
    numBeamsWeWant = params.numBeamsWeWant;
    ChangeWeightsTrigger = params.ChangeWeightsTrigger;
    reductionFactor = .5; % This is a line search parameter.
    beamWeightsInit = params.beamWeightsInit;
    beamWeight = params.beamWeight;
    beamWeights = beamWeight*beamWeightsInit;
    beamSizes = params.beamSizes;
    numBeams = length(beamSizes);
    numBeamsTotal = numBeams;
    beamNumbers = 1:numBeams;
    BeamletLog0 = params.BeamletLog0;
    BeamletLog = (BeamletLog0);
    maskBeamletLog = BeamletLog==1;
    numBeamletsPerB = size(BeamletLog0,1)*size(BeamletLog0,2);
    
    maxDose = Weights.maxDose;
    maxWeightsLong = Weights.maxWeightsLong;
    minDoseTarget = Weights.minDoseTarget;
    minDoseTargetWeights = Weights.minDoseTargetWeights;
    OARWeightsLong = Weights.OARWeightsLong;
    numVoxPTV = length(minDoseTarget);
    
    DTrans = D';

    % To add a buffer to contain dimension reduction results
    dimensionReductionTime = {};
    
    numBeamlets = size(A,2); %numBeamlets is never reduced
    topN = cell(maxIter,1);
    
        function [grad,cost] = eval_grad(x)
            
            Ax = ATrans'*x;
            prox1 = min(Ax(1:numVoxPTV) - minDoseTarget,0);
            prox2 = max(Ax - maxDose,0);
            term3 = Ax(numVoxPTV+1:end);
            term4 = DTrans'*x;
            prox4 = prox1Norm(term4,gamma);
    
            grad = A'*([minDoseTargetWeights.*(prox1);OARWeightsLong.*term3] ...
                + maxWeightsLong.*(prox2)) + eta*(D'*(term4 - prox4)/gamma);
            cost = .5*sum(minDoseTargetWeights.*((prox1).^2)) ...
                + .5*sum(maxWeightsLong.*((prox2).^2)) + .5*sum(OARWeightsLong.*(term3.^2))...
                + eta*(sum(abs(prox4)) + (.5/gamma)*sum((prox4 - term4).^2));
    
        end
    
        function cost = eval_g(x) % This function computes gx without extra expense of computing grad_gx.
            
            Ax = ATrans'*x;
            prox1 = min(Ax(1:numVoxPTV) - minDoseTarget,0);
            prox2 = max(Ax - maxDose,0);
            term3 = Ax(numVoxPTV+1:end);
            term4 = DTrans'*x;
            prox4 = prox1Norm(term4,gamma);        
            
            cost = .5*sum(minDoseTargetWeights.*((prox1).^2)) ...
                + .5*sum(maxWeightsLong.*((prox2).^2)) + .5*sum(OARWeightsLong.*(term3.^2))...
                + eta*(sum(abs(prox4)) + (.5/gamma)*sum((prox4 - term4).^2));
    
        end
    
    xkm1 = rand(numBeamlets,1);
    vkm1 = xkm1;
    disp(['all zero cost is: ',num2str(eval_g(0*xkm1))])
    numBeamslast = numBeams;
    costs = zeros(maxIter,1);
    
    for k = 1:maxIter
        
        if (k <= 50 || mod(k,5) == 0)
            t = t/reductionFactor; % Attempt to increase t.
        end
        accept_t = 0;
        while accept_t == 0
            if k > 1
                a = tkm1;
                b = t*theta_km1^2;
                c = -t*theta_km1^2;
                
                theta = (-b + sqrt(b^2 - 4*a*c))/(2*a);
                y = (1 - theta)*xkm1 + theta*vkm1;
            else
                theta = 1;
                y = xkm1;
            end
            [gradAty,gy] = eval_grad(y);
            
            in = y - t*gradAty;
            xfull = 0*BeamletLog;
            xfull(maskBeamletLog) = max(in,0);
            x2d = reshape(xfull,[numBeamletsPerB,numBeams]);
            [x2dprox,nrm] = proxL2Onehalf_QL_cpu(x2d, t*beamWeights);
            x = x2dprox(maskBeamletLog);
    
            gx = eval_g(x);
            lhs = gx;
            rhs = gy + gradAty'*(x - y) + (.5/t)*sum((x-y).^2);
            if lhs <= rhs
                accept_t = 1;
            else
                t = reductionFactor*t;
            end
            
        end
        
        v = xkm1 + (1/theta)*(x - xkm1);
        
        theta_km1 = theta;
        tkm1 = t;
        xkm1 = x;
        vkm1 = v;
        
        %%%%%%%%%%%% Now compute objective function value. %%%%%%%%%%%%%%%%%%
        beamNorms = nrm(:);
        cost = gx + sum(beamWeights.*sqrt(beamNorms));
        costs(k) = cost;
        
        numActiveBeams = nnz(beamNorms > 1e-2);
        activeBeamsStrict = find(beamNorms > 1e-6);
        numActiveBeamsStrict = numel(activeBeamsStrict);
        nonactiveBeamsStrict = beamNorms <= 1e-6;
        
        [sortedNorms,sortedBeams] = sort(beamNorms,'descend');
        activeBeams = beamNumbers(sortedBeams(1:numActiveBeams));
        activeNorms = sortedNorms(1:numActiveBeams);
        topN_k = beamNumbers(sortedBeams(1:min(20,numActiveBeams)));
        topN{k} = topN_k;
        
        %%%%%%%%%%%%% Finished computing objective function value.
        if mod(k,ChangeWeightsTrigger)==0
            if(numActiveBeamsStrict >= 2*numBeamsWeWant)
                beamWeights = beamWeights*2;
            elseif(numActiveBeamsStrict >= 1.5*numBeamsWeWant)
                beamWeights = beamWeights*1.5;
            elseif(numActiveBeamsStrict >= 1.05*numBeamsWeWant)
                beamWeights = beamWeights*1.2;
            elseif(numActiveBeamsStrict < numBeamsWeWant)
                beamWeights = beamWeights/3;
            end
        end
        
        %%%%%%%%% Now throw out inactive beams
        if (mod(k,pruneTrigger) == 0 & numBeamslast > numActiveBeamsStrict + 20)
            dimensionReductionTic = tic;
            BeamletInd = BeamletLog;
            BeamletInd(maskBeamletLog) = 1:nnz(BeamletLog);        
            BeamletInd(:,:,nonactiveBeamsStrict)=0;
            beamletList = BeamletInd(maskBeamletLog);
            beamletList = beamletList(beamletList~=0);
            
            A = A(:,beamletList);
            D = D(:,beamletList);
            
            ATrans = A';
            DTrans = D';
                            
            BeamletLog = BeamletLog0;
            BeamletLog(:,:,nonactiveBeamsStrict) = 0;
            maskBeamletLog = (BeamletLog==1);
            
            xkm1 = xkm1(beamletList);
            vkm1 = vkm1(beamletList);
    
            numBeamslast = numActiveBeamsStrict;
            dimensionReductionTimeElapsed = toc(dimensionReductionTic);
            dimensionReductionTime{end + 1} = dimensionReductionTimeElapsed;
        end
        %%%%%%%%%% Finished throwing out inactive beams
    
        if mod(k,showTrigger) == 0
            disp(['FISTA iteration is: ', num2str(k),'cost: ',num2str(cost),' t: ',num2str(t),' numActiveBeams: ',num2str(numActiveBeams),' Top beams: ', num2str(topN_k)])
            disp(['numActiveBeamsStrict: ',num2str(numActiveBeamsStrict)])
            beamNormsAll = zeros(numBeamsTotal,1);
            beamNormsAll(activeBeams) = activeNorms;
            
            figure(100)
            plot(beamNormsAll)
            drawnow
            
            figure(200)
            semilogy(costs);
            
        end
        
        if(numActiveBeamsStrict <= numBeamsWeWant)
            break
        elseif(numActiveBeamsStrict <= numBeamsWeWant+1)
            if(abs(costs(k)-costs(k-1))<1e-05*costs(k))
    %             break
            end
        elseif(numActiveBeamsStrict <= numBeamsWeWant*1.05)
            if(abs(costs(k)-costs(k-1))<1e-07*costs(k))
                break
            end
        end
            
    end
    
    xFull = zeros(numBeamlets,1);
    xFull(maskBeamletLog) = xkm1;
    
end
    