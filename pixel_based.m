addpath('GCMex2.3.0');
addpath('CommonFunc');
close all;
clearvars

%% read an image
im = im2double(imread('figure\cow.jpg'));
im = imresize(im, 1.0);

%% hyperparameters
lambda  = 10;

%% k-means clustering
k = 4;
distance = 'sqEuclidean';
data = ToVector(im);
[idx, c] = kmeans(data, k, 'distance', distance,'maxiter', 200);


%% data term
tic; fprintf('data term: ');
[H,W,D] = size(im);
N = H*W;
imagePixel = reshape(im,[N 3]);

for i = 1:k
    likelihood(:,:,i) = sqrt(sum((imagePixel - repmat(c(i,:),[N 1])).^2,2));
end

unary = [];
for i = 1:k
    unary = [unary, likelihood(:,:,i)];
end
unary = unary';
toc;

%% smoothness term
tic; fprintf('smoothness term: ');
pairwise = sparse(N,N);

% add all horizontal links
for x = 1:W-1
  for y = 1:H
    node  = 1 + (y-1) + (x-1)*H;
    right = 1 + (y-1) + x*H;
    distance = norm(imagePixel(node,:) - imagePixel(right,:));
    pairwise(node,right) = distance;
    pairwise(right,node) = distance;
  end
end

% add all vertical nbr links
for x = 1:W
  for y = 1:H-1
    node = 1 + (y-1) + (x-1)*H;
    down = 1 + y + (x-1)*H;
    distance = norm(imagePixel(node,:) - imagePixel(down,:));
    pairwise(node,down) = distance;
    pairwise(down,node) = distance;
  end
end
toc;

%% graphcut
Sc = ones(k) - eye(k);
labelcost = Sc*lambda;
segclass = zeros(N,1);

[labels E Eafter] = GCMex(segclass, single(unary), pairwise, single(labelcost), 0);

figure;
subplot(121);
imagesc(im);
axis image; title('Input image');

subplot(122);
imagesc(reshape(labels,[H W]));
axis image; title('Graphcut (Pixel-based)');

