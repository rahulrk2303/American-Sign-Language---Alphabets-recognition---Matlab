clear;
%%
load MyTrainedNN.mat;
net = MyTrainedNN;

%%
[file,path]=uigetfile({'*.jpg;*.bmp;*.png;*.tif'},'Choose an image');
s=[path,file];
I=imread(s);

%%
I = imresize(I,[227,227]);

%%
image(I);

%%
label = classify (net, I)

%%
title(char(label));

%%
tts(char(label));
