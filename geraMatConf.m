function isLabels = geraMatConf(respostas, predito, medidas)
%GERAMATCONF Funcao que vai gerar e imprimir a matriz de confusao para os
%dados passados como parametro para a funcao

%% Geracao dos dados necessarios para a criacao da matriz de confusao

% Passos para a criacao da matriz de confusao
isLabels = unique(respostas);
nLabels = numel(isLabels);
[n,p] = size(medidas);

[~,grpLabel] = ismember(predito,isLabels); 
labelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpLabel,(1:n)'); 
labelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(respostas,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1;

figure;
plotconfusion(YMat, labelMat);
pause;

end

