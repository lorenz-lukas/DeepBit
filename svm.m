%% Projeto final de FSI - 1/2018
%Lukas Lorenz de Andrade - 
%Victor Araujo Vieira - 14/0032801

%% Script que executa os comandos necessarios para aplicar o modelo deepbit treinado nas imagens de insetos
% Sera dividido em 4 partes: Inicializacao dos dados, extracao dos
% descritores binarios para todas imagens,
% treinamento dos SVM e, por ultimo, o teste da eficiencia dos classificadores.

%% Inicializacao dos dados
% Parte do script que vai preparar os dados

close all;
clear all;

addpath(genpath(pwd));

% variaveis do caffe
addpath('../cvpr16-deepbit/matlab');
% Mude firsttime para 1, caso seja a primeira vez que esteja rodando o
% codigo
firstTime = 1;
% Mude modelOption para 1 se quiser usar o modelo que gera 16 bits, 2 32
% bits e 3 64 bits. Por default eh 32
modelOption = 2;
switch modelOption
        case 1
            % modelo deepbit
            model_file = '../cvpr16-deepbit/models/deepbit/DeepBit16_final_iter_1.caffemodel';
            % definicao do modelo
            model_def_file = '../cvpr16-deepbit/models/deepbit/deploy16.prototxt';
            feat_result_file = sprintf('resultDeepBitCrop16.mat', 'resultDeepBitCrop');
        case 3
            % modelo deepbit
            model_file = '../cvpr16-deepbit/models/deepbit/DeepBit64_final_iter_1.caffemodel';
            % definicao do modelo
            model_def_file = '../cvpr16-deepbit/models/deepbit/deploy64.prototxt';
            feat_result_file = sprintf('resultDeepBitCrop64.mat', 'resultDeepBitCrop');
        otherwise
            % modelo deepbit
            model_file = '../cvpr16-deepbit/models/deepbit/DeepBit32_final_iter_1.caffemodel';
            % definicao do modelo
            model_def_file = '../cvpr16-deepbit/models/deepbit/deploy32.prototxt';
            feat_result_file = sprintf('resultDeepBitCrop32.mat', 'resultDeepBitCrop');
    end

if(firstTime == 1)
    
    caffe.set_mode_gpu();
    caffe.set_device(0);
    net = caffe.Net(model_def_file, model_file, 'test');
    net.blobs('data').reshape([224 224 3 1]); % reshape blob 'data'

    mediaBin = 0; % media que vai ser usada para a binarizacao dos atributos extraidos pelo deepbit

    todasImagens = './images.txt';
    listaImagens = read_cell(todasImagens);

    %% Extracao dos descritores binarios para todas as imagens e calculo da media mediaBin

    numImagens = length(listaImagens);

    % Loop que vai ler as imagens da lista de imagens e fazer as operacoes
    % necessarias
    
    IMAGE_DIM = 256;
    CROPPED_DIM = 224;

    % Se for a primeira vez que esta rodando, faz todo o procedimento, se nao for,
    % ja carrega o objeto mat
    % Para cada imagem, vai rodar o modelo deepbit e vai adicinar o resultado
    % em um vetor coluna, de modo que vai ficar 4900x32
   
    indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
    center = floor(indices(2) / 2)+1;
    for i = 1:numImagens
        im = imread(listaImagens{i});
        im = imresize(im, [256, 256], 'bilinear');
        im = permute(im(center:center+CROPPED_DIM-1,...
        center:center+CROPPED_DIM-1,:),[2 1 3]);
        resultModel = net.forward({im});
        resultDeepBitCrop(i, :) = resultModel{1, 1};
    end
    save(feat_result_file);

else
    load(['./' feat_result_file]);
    %load('./resultDeepBitCrop.mat');
end

% Calcula a media e mediana de cada coluna
mediaCol = mean(resultDeepBitCrop);
medianCol = median(resultDeepBitCrop);
% Calcula o resultado da media e mediana de cada coluna, ou seja, calcula
% agora as medias e mediana gerais
mediaBin = mean(mediaCol);
medianBin = median(medianCol);

% Avalia os features originais de cada img, se for maior que a media geral
% vira 1, senao 0
binarioImagensMedia = (resultDeepBitCrop > mediaBin);
binarioImagensMedia = double(binarioImagensMedia); % converte de logical para double

% Avalia os features originais de cada img, se for maior que a mediana geral
% vira 1, senao 0
binarioImagensMediana = (resultDeepBitCrop > medianBin);
binarioImagensMediana = double(binarioImagensMediana); % converte de logical para double

%% Treinamento da SVM com cross validation e usando media

% Le o arquivo que contem as labels de todas as classes
classes = read_cell('./labels.txt');

% Usando o resultado ao avaliar todas as imagens no modelo DeepBit
% Ao inves de usar os descritores binarios 
rng(1)
disp('Treino com media');

options = statset('UseParallel',true);
t = templateSVM('KernelFunction', 'gaussian'); % normaliza os dados
%t = templateKNN('NumNeighbors',5, 'Distance', 'mahalanobis');
%SVMClassifierMedia = fitcecoc(binarioImagensMedia, classes, 'Coding', 'onevsall', 'Learners', t, 'Options', options);
tic
SVMClassifierMedia = fitcecoc(binarioImagensMedia, classes, 'Coding', 'onevsall', 'Learners', t);
toc

tic
CVClassifier = crossval(SVMClassifierMedia);
toc

estErr1 = kfoldLoss(CVClassifier)

%% Teste da eficiencia dos classificadores
% testando classificador com cross validation
resultCV = kfoldPredict(CVClassifier);

%% Treinamento da SVM com cross validation e usando mediana
disp('Treino com mediana');

options = statset('UseParallel',true);
t = templateSVM('KernelFunction', 'gaussian'); % normaliza os dados
%t = templateKNN('NumNeighbors',50, 'Distance', 'cosine');
% SVMClassifierMediana = fitcecoc(binarioImagensMediana, classes, 'Coding', 'onevsall', 'Learners', t, 'Options', options);
tic
SVMClassifierMediana = fitcecoc(binarioImagensMediana, classes, 'Coding', 'onevsall', 'Learners', t);
toc

tic
CVClassifierMediana = crossval(SVMClassifierMediana);
toc

estErr2 = kfoldLoss(CVClassifierMediana)
resultCVMediana = kfoldPredict(CVClassifierMediana);

