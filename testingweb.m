    clear;
    camera = webcam;
    load MyTrainedNN.mat;
    net = MyTrainedNN;
    bg = camera.snapshot;
    
    while true
        I = camera.snapshot;
        I = imresize(I,[227,227]);
        image(I);
        label = classify (net, I);
        title(char(label));
        drawnow;
        tts(char(label));
    end