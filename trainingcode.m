%% Load the Data set
allImages = imageDatastore('asl_alphabet_train', ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Split data into training and test sets 
[trainingImages, testImages] = ...
    splitEachLabel(allImages, 0.8, 'randomize');
 
%% Load Pre-trained Network (AlexNet)
alex = alexnet; 

%% Review Network Architecture 
layers = alex.Layers

%% Modify Pre-trained Network 
layers(23) = fullyConnectedLayer(29); 
layers(25) = classificationLayer;

%% Perform Transfer Learning
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.0001,...
    'MaxEpochs', 3, 'MiniBatchSize', 32, 'plot', 'training-progress');

%% Set custom read function 
trainingImages.ReadFcn = @readFunctionTrain;

%% Train the Network  
myNet = trainNetwork(trainingImages, layers, opts);

%% Save the Neural Network for Testing
MyTrainedNN = myNet;
save MyTrainedNN;

%% Test Network Performance
testImages.ReadFcn = @readFunctionTrain;
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)
