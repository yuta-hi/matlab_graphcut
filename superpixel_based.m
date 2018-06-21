addpath(genpath('GCMex2.3.0'));
addpath('CommonFunc');
addpath('SLIC');
close all;
clearvars

%% read an image
im = im2double(imread('figure\cow.jpg'));
im = imresize(im, 1.0);

%% hyperparameters
lambda = 3.0;
sigma = 2;

%% k-means clustering
k = 4;
distance = 'sqEuclidean';
data = ToVector(im);
[idx, c] = kmeans(data, k, 'distance', distance,'maxiter', 200);

%% superpixel
[l, Am, C] = slic(im, 1000, 10, 1, 'median');
connectivity = 8;
[Am, varargout] = regionadjacency(l, connectivity);

figure;
subplot(121);
imagesc(drawregionboundaries(l, im, [255]));
axis image; title('Input image');
subplot(122); 
imagesc(drawregionboundaries(l, zeros(size(im)), [255, 255, 0]));
axis image; title('Input image');

%% data term
tic; fprintf('data term: ');
[H,W,D] = size(im);
N = H*W;
imagePixel = reshape(im,[N 3]);

% pixel-based
for i = 1:k
    icv = inv(cov(data(idx==i,:)));    
    dif = imagePixel - repmat(c(i,:),[N 1]);
    likelihood(:,:,i) = sum((dif*icv).*dif./2,2);
end

unary = [];
for i = 1:k
    unary = [unary, likelihood(:,:,i)];
end
unary = unary';

% superpixel-based
unary_spixel = zeros(k, size(Am,1));
for i = 1:size(Am,1)
    label = (l==i);
    unary_spixel_tmp = unary(:, find(label));
    unary_spixel(:,i) = mean(unary_spixel_tmp, 2);
end
toc;


%% smoothness term
tic; fprintf('smoothness term: ');

diff_image = zeros([size(im), 4]); % 4 direction
for i=1:size(im,3)
    % dir x
    s = [0 1 0; 0 0 0; 0 -1 0];
    H = conv2(im(:,:,i), s, 'same');
    diff_image(:,:,i,1) = power(H, 2.0);
    % dir y
    s = s';
    H = conv2(im(:,:,i), s, 'same');
    diff_image(:,:,i,2) = power(H, 2.0);
    % dir left-up
    s = [1 0 0; 0 0 0; 0 0 -1];
    H = conv2(im(:,:,i), s, 'same');
    diff_image(:,:,i,3) = power(H, 2.0);
    % dir right-up
    s = [0 0 1; 0 0 0; -1 0 0];
    H = conv2(im(:,:,i), s, 'same');
    diff_image(:,:,i,4) = power(H, 2.0);
end

[w,h,c,d] = size(diff_image);
norm_image = zeros(w,h,d);
for j=1:d
    for i=1:c
        norm_image(:,:,j) = norm_image(:,:,j) + diff_image(:,:,i,j);
    end
    norm_image(:,:,j) = sqrt(norm_image(:,:,j));
end
norm_image = min(norm_image,[],3);
%norm_image = max(norm_image,[],3);
norm_image_vec = reshape(norm_image, 1, []);

pairwise = sparse(size(Am));

for i = 1:size(Am,1)
    
    [~,connect_indices] = find(Am(i, :));

    for j = 1:length(connect_indices)

        se = strel('square',2);
        source = imdilate((l==i), se);
        target = imdilate((l==connect_indices(j)), se);
        contour = (source+target==2);
        
        smooth_cost = norm_image_vec(find(contour));
        smooth_cost = exp(-mean(smooth_cost)./ power(sigma, 2.0));
        %smooth_cost = mean(smooth_cost);

        pairwise(i, connect_indices(j)) = smooth_cost;
        pairwise(connect_indices(j), i) = smooth_cost;
    end
end
toc;

%% graphcut
N = size(Am,1);
Sc = ones(k) - eye(k);
labelcost = Sc*lambda;
segclass = zeros(N,1);

[labels, E, Eafter] = GCMex(segclass, single(unary_spixel), pairwise, single(labelcost), 0);

estimated_label = nan(size(im,1), size(im,2));

for pixel = 1:N
    estimated_label(l == pixel) = labels(pixel);
end

figure;
subplot(121);
imagesc(im);
axis image; title('Input image');

subplot(122);
imagesc(estimated_label);
axis image; title('Graphcut (Superpixel-based)');

