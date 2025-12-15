% batch_process_smk.m
clc; clear; close all;

addpath(genpath('./'));

% Configuration
input_dir = './data/dataset_motion_blury/';
output_dir = './data/dataset_SMK_Restored/';

if ~exist(output_dir, 'dir'); mkdir(output_dir); end

% Parameters (Matched to our dataset generation settings)
opts.kernel_size = 21;  % Slightly larger than motion degree (15)
opts.gamma_correct = 1.0;
opts.lambda_tv = 0.003;
opts.lambda_l0 = 2e-3;

img_files = dir(fullfile(input_dir, '*.jpg')); 

for i = 1:length(img_files)
    filename = img_files(i).name;
    img_path = fullfile(input_dir, filename);
    
    fprintf('Processing %s ...\n', filename);
    
    try
        y = im2double(rgb2gray(imread(img_path)));
        % Call the core blind deconvolution function from the repo
        [kernel, latent_img] = blind_deconv_main(y, opts); 
        imwrite(latent_img, fullfile(output_dir, filename));
    catch
        fprintf('Failed to process %s\n', filename);
    end
end