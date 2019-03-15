clear;
camera = webcam;
load MyTrainedNN.mat;
net = MyTrainedNN;
%while true
    
    IM1 = camera.snapshot;
    figure; imshow(IM1); title('Background');
    pause(3);

    IM2= camera.snapshot;
    figure; imshow(IM2); title('Gesture');
    IM3 = IM1 - IM2;
    %figure(1);subplot(3,2,3);imshow(IM3);title('Subtracted');
    IM3 = rgb2gray(IM3);   
    figure; imshow(IM3);%Converts RGB to Gray
   % figure(1);subplot(3,2,4);imshow(IM3);title('Grayscale');                    %Display Gray Image
    lvl = graythresh(IM3);
    IM3 = im2bw(IM3, lvl); 
   % figure(1);subplot(3,2,5);imshow(IM3);title('Black&White'); 
    IM3 = bwareaopen(IM3, 10000);
IM3 = imfill(IM3,'holes');
IM3 = imerode(IM3,strel('disk',15));                                        %erode image
IM3 = imdilate(IM3,strel('disk',20));                                       %dilate iamge
IM3 = medfilt2(IM3, [5 5]); 
IM3 = bwareaopen(IM3, 10000);  
figure; imshow(IM3);
%figure(1);subplot(3,2,6);imshow(IM3);title('Small Areas removed & Holes Filled');  
   
  %  IM3 = cat(3, IM3, IM3, IM3);
    figure; imshow(IM3); title('Black&White');
    
    gray = rgb2gray(IM2);
    gray(~IM3) = 255;
    imshow(gray);
    title ('Final');
        
    r = IM2(:,:,1);
    g = IM2(:,:,2);
    b = IM2(:,:,3);
    r(~IM3) = 255;
    g(~IM3) = 255;
    b(~IM3) = 255;
    IM = cat(3,r,g,b);
    imshow(IM);
    I = imresize(IM,[227,227]);
    image(IM2);
    label = classify (net, I)
    title(char(label));
    
    %tts(char(label));
%end