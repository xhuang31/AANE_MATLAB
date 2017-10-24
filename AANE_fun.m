function H = AANE_fun(Net,Attri,d,varargin)
%Jointly embed Net and Attri into embedding representation H
%     H = AANE_fun(Net,Attri,d);
%     H = AANE_fun(Net,Attri,d,lambda,rho);
%     H = AANE_fun(Net,Attri,d,lambda,rho,'Att');
%     H = AANE_fun(Net,Attri,d,lambda,rho,'Att',worknum);
% 
%         Net   is the weighted adjacency matrix
%        Attri  is the attribute information matrix with row denotes nodes
%         d     is the dimension of the embedding representation
%       lambda  is the regularization parameter
%         rho   is the penalty parameter
%        'Att'  refers to conduct Initialization from the SVD of Attri
%       worknum is the number of worker used for distribution

%   Copyright 2017, Xiao Huang and Jundong Li.
%   $Revision: 1.0.0 $  $Date: 2017/10/18 00:00:00 $

%% Parameters
maxIter = 2; % Max num of iteration
n = size(Net,1); % Total number of nodes
Net(1:n+1:n^2) = 0;
lambda = 0.1; % Initial regularization parameter
rho = 5; % Initial penalty parameter
Block = min(ceil(n/1),7575); % Treat each 7575 nodes as a block
if ~isempty(varargin)
    lambda = varargin{1};
    rho = varargin{2};
    if length(varargin) >= 3 && strcmp(varargin{3},'Att')
        [~,MaxEdges] = sort(sum(Attri),'descend');
        [H,~] = svds(Attri(:,MaxEdges(1:min(10*d,size(Attri,2)))),d);
    else
        [~,MaxEdges] = sort(sum(Net),'descend');
        [H,~] = svds(Net(:,MaxEdges(1:min(10*d,n))),d);
    end
    if length(varargin) >= 4
        Block = min(ceil(n/varargin{4}),7575); % Treat each 7575 nodes as a block
    end
else
    [~,MaxEdges] = sort(sum(Net),'descend');
    [H,~] = svds(Net(:,MaxEdges(1:min(10*d,n))),d);
end
Z = diag(sum(Attri.^2,2).^-.5); % temporary value
Z(isinf(Z)) = 0; % temporary value
Attri = Attri'*Z; % Normalization
%% First update H
Z = H'; % Transpose for speedup
XTX = Z*H*2; % Transpose for speedup
H = Z; % Transpose for speedup
for Blocki = 1:ceil(n/Block) % Split nodes into different Blocks
    IndexBlock = 1+Block*(Blocki-1); % First Index for splitting blcks
    LocalIndex = IndexBlock:IndexBlock-1+min(n-IndexBlock+1,Block);
    SA = Attri(:,LocalIndex)'*Attri; % Local affinity matrix for Blocki
    sumS = Z*SA'*2;
    for i = LocalIndex 
        Neighbor = Z(:,Net(:,i)~=0); % the set of adjacent nodes of node i
        for j = 1:1
            normi_j = sum((bsxfun(@minus,Neighbor,H(:,i))).^2).^.5; % norm of h_i^k-z_j^k
            nzIdx = logical(normi_j); % Non-equal Index
            if any(nzIdx)
                Wij = lambda*Net(i,Net(:,i)~=0);
                normi_j = Wij(nzIdx)./normi_j(nzIdx);
                H(:,i)=(XTX+(sum(normi_j)+rho)*eye(d))\(sumS(:,i-IndexBlock+1)+sum(bsxfun(@times,Neighbor(:,nzIdx),normi_j),2)+rho*Z(:,i));
            else
                H(:,i)=(XTX+rho*eye(d))\(sumS(:,i-IndexBlock+1)+rho*Z(:,i));
            end
        end
    end
end
Affi = Blocki; % Index for affinity matrix
U = zeros(d,n);
%% Iterations
for iter = 1:maxIter-1
    %% Update Z
    XTX = H*H'*2;
    for Blocki = 1:ceil(n/Block) % Split nodes into different Blocks
        IndexBlock = 1+Block*(Blocki-1); % Index for splitting blcks
        LocalIndex = IndexBlock:IndexBlock-1+min(n-IndexBlock+1,Block);
        if Affi ~= Blocki
            SA = Attri(:,LocalIndex)'*Attri;
            Affi = Blocki; % Index for affinity matrix
        end
        sumS = H*SA'*2;
        for i = LocalIndex
            Neighbor = H(:,Net(:,i)~=0); % the set of adjacent nodes of node i
            for j = 1:1
                normi_j = sum((bsxfun(@minus,Neighbor,Z(:,i))).^2).^.5;
                nzIdx = logical(normi_j); % Non-equal Index
                if any(nzIdx)
                    Wij = lambda*Net(i,Net(:,i)~=0);
                    normi_j = Wij(nzIdx)./normi_j(nzIdx);
                    Z(:,i)=(XTX+(sum(normi_j)+rho)*eye(d))\(sumS(:,i-IndexBlock+1)+sum(bsxfun(@times,Neighbor(:,nzIdx),normi_j),2)+rho*(H(:,i)+U(:,i)));
                else
                    Z(:,i)=(XTX+rho*eye(d))\(sumS(:,i-IndexBlock+1)+rho*(H(:,i)+U(:,i)));
                end
            end
        end
    end
    U = U+H-Z; % Update U
    %% Update H
    XTX = Z*Z'*2; % Transposed for speedup
    for Blocki = 1:ceil(n/Block) % Split nodes into different Blocks
        IndexBlock = 1+Block*(Blocki-1); % Index for splitting blcks
        LocalIndex = IndexBlock:IndexBlock-1+min(n-IndexBlock+1,Block);
        if Affi ~= Blocki
            SA = Attri(:,LocalIndex)'*Attri;
            Affi = Blocki; % Index for affinity matrix
        end
        sumS = Z*SA'*2;
        for i = LocalIndex
            Neighbor = Z(:,Net(:,i)~=0); % the set of adjacent nodes of node i
            for j = 1:1
                normi_j = sum((bsxfun(@minus,Neighbor,H(:,i))).^2).^.5;
                nzIdx = logical(normi_j); % Non-equal Index
                if any(nzIdx)
                    Wij = lambda*Net(i,Net(:,i)~=0);
                    normi_j = Wij(nzIdx)./normi_j(nzIdx);
                    H(:,i)=(XTX+(sum(normi_j)+rho)*eye(d))\(sumS(:,i-IndexBlock+1)+sum(bsxfun(@times,Neighbor(:,nzIdx),normi_j),2)+rho*(Z(:,i)-U(:,i)));
                else
                    H(:,i)=(XTX+rho*eye(d))\(sumS(:,i-IndexBlock+1)+rho*(Z(:,i)-U(:,i)));
                end
            end
        end
    end
end
H = H'; % H is transposed
