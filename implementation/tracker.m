function results = tracker(params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Correct max number of samples
params.nSamples = min(params.nSamples, seq.num_frames);

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    warning('ECO:tracker', 'Error when using the mexResize function. Using Matlab''s interpolation function instead, which is slower.\nTry to run the compile script in "external_libs/mexResize/".\n\nThe error was:\n%s', getReport(err));
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2]; % for testing
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
feature_dim = feature_info.dim;
num_feature_blocks = length(feature_dim);

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Set the sample feature dimension
sample_dim = feature_dim;

% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% Number of Fourier coefficients to save for each filter layer. This will
% be an odd number.
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size.
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% How much each feature block has to be padded to the obtain output_sz
pad_sz = cellfun(@(filter_sz) (output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false);

% Compute the Fourier series indices and their transposes
ky = cellfun(@(sz) (-ceil((sz(1) - 1)/2) : floor((sz(1) - 1)/2))', filter_sz_cell, 'uniformoutput', false);
kx = cellfun(@(sz) -ceil((sz(2) - 1)/2) : 0, filter_sz_cell, 'uniformoutput', false);

% construct the Gaussian label function using Poisson formula
sig_y = sqrt(prod(floor(base_target_sz))) * params.output_sigma_factor * (output_sz ./ img_support_sz);
yf_y = cellfun(@(ky) single(sqrt(2*pi) * sig_y(1) / output_sz(1) * exp(-2 * (pi * sig_y(1) * ky / output_sz(1)).^2)), ky, 'uniformoutput', false);
yf_x = cellfun(@(kx) single(sqrt(2*pi) * sig_y(2) / output_sz(2) * exp(-2 * (pi * sig_y(2) * kx / output_sz(2)).^2)), kx, 'uniformoutput', false);
yf = cellfun(@(yf_y, yf_x) cast(yf_y * yf_x, 'like', params.data_type), yf_y, yf_x, 'uniformoutput', false);

% construct cosine window
cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

% Compute Fourier series of interpolation function
[interp1_fs, interp2_fs] = cellfun(@(sz) get_interp_fourier(sz, params), filter_sz_cell, 'uniformoutput', false);

% Get the reg_window_edge parameter
reg_window_edge = {};
for k = 1:length(features)
    if isfield(features{k}.fparams, 'reg_window_edge')
        reg_window_edge = cat(3, reg_window_edge, permute(num2cell(features{k}.fparams.reg_window_edge(:)), [2 3 1]));
    else
        reg_window_edge = cat(3, reg_window_edge, cell(1, 1, length(features{k}.fparams.nDim)));
    end
end

% Construct spatial regularization filter
reg_filter = cellfun(@(reg_window_edge) get_reg_filter(img_support_sz, base_target_sz, params, reg_window_edge), reg_window_edge, 'uniformoutput', false);

[nScales, scale_step, scaleFactors, scale_window, scale_model_sz, params, ysf] = init_scale_filter(params);

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

if params.use_hw_scale_filter
    current_hw_ScaleFactor=[1,1];
    [~, ~, hw_scaleFactors, hw_scale_window, params, hw_ysf] = init_hw_scale_filter(params);
end
% end

% Set conjugate gradient uptions
% init_CG_opts.CG_use_FR = true;
% init_CG_opts.tol = 1e-6;
% init_CG_opts.CG_standard_alpha = true;
% init_CG_opts.debug = params.debug;
CG_opts.CG_use_FR = params.CG_use_FR;
CG_opts.tol = 1e-6;
CG_opts.CG_standard_alpha = params.CG_standard_alpha;
CG_opts.debug = params.debug;
if params.CG_forgetting_rate == Inf || params.learning_rate >= 1
    CG_opts.init_forget_factor = 0;
else
    CG_opts.init_forget_factor = (1-params.learning_rate)^params.CG_forgetting_rate;
end

seq.time = 0;

% Initialize and allocate
prior_weights = zeros(params.nSamples,1, 'single');
sample_weights = cast(prior_weights, 'like', params.data_type);
samplesf = cell(1, 1, num_feature_blocks);
samplesf1 = cell(1, 1, num_feature_blocks);
samplesf2 = cell(1, 1, num_feature_blocks);
samplesf3 = cell(1, 1, num_feature_blocks);
samplesf4 = cell(1, 1, num_feature_blocks);
% samplesf_single = cell(1, 1, num_feature_blocks);
if params.use_gpu
    % In the GPU version, the data is stored in a more normal way since we
    % dont have to use mtimesx.
    for k = 1:num_feature_blocks
        samplesf{k} = zeros(filter_sz(k,1),(filter_sz(k,2)+1)/2,sample_dim(k),params.nSamples, 'like', params.data_type_complex);
    end
else
    for k = 1:num_feature_blocks
        samplesf{k} = zeros(params.nSamples,sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
        samplesf1{k} = zeros(params.nSamples,sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
        samplesf2{k} = zeros(params.nSamples,sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
        samplesf3{k} = zeros(params.nSamples,sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
        samplesf4{k} = zeros(params.nSamples,sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
        %         samplesf_single{k} = zeros(params.nSamples,sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
    end
end

% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);

% % Distance matrix stores the square of the euclidean distance between each pair of
% % samples. Initialise it to inf
% distance_matrix = inf(params.nSamples, 'single');

% Kernel matrix, used to update distance matrix
% gram_matrix = inf(params.nSamples, 'single');

latest_ind = [];
frames_since_last_train = inf;
% num_training_samples = 0;

% Find the minimum allowed sample weight. Samples are discarded if their weights become lower
params.minimum_sample_weight = params.learning_rate*(1-params.learning_rate)^(2*params.nSamples);

res_norms = [];
residuals_pcg = [];

while true
    
    train_tracker = (seq.frame < params.skip_after_frame) || (frames_since_last_train >= params.train_gap);
    
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        if ~params.use_scale
            scaleFactors = 1;
        end
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
%             det_sample_pos = sample_pos;
            xt = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
            
            % Do windowing of features
            xt_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf_proj = cellfun(@cfft2, xt_proj, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            xtf_proj = interpolate_dft(xtf_proj, interp1_fs, interp2_fs);
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = sum(bsxfun(@times, hf_full{k1}, xtf_proj{k1}), 3);
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds
                scores_fs_feat{k} = sum(bsxfun(@times, hf_full{k}, xtf_proj{k}), 3);
                scores_fs_sum(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) = ...
                    scores_fs_sum(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) + ...
                    scores_fs_feat{k};
            end
            
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            
            % Optimize the continuous score function with Newton's method.
            [trans_row, trans_col, scale_ind] = optimize_scores(scores_fs, params.newton_iterations);
            
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz) * currentScaleFactor;
            %                 scale_change_factor = scaleFactors(scale_ind);
            
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            % Scale
            if params.use_scale_filter
                xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
                xsf = fft(xs,[],2);
                scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + params.scale_lambda) ));
                % find the maximum scale response
                recovered_scale = find(scale_response == max(scale_response(:)), 1);
                % update the scale
                currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
                if currentScaleFactor < min_scale_factor
                    currentScaleFactor = min_scale_factor;
                elseif currentScaleFactor > max_scale_factor
                    currentScaleFactor = max_scale_factor;
                end
            end
            if params.use_hw_scale_filter
                hw_xs=get_scale_hw_sample(im, pos, base_target_sz.*current_hw_ScaleFactor, currentScaleFactor,hw_scaleFactors, hw_scale_window, scale_model_sz);
                hw_xsf=fft(hw_xs,[],2);
                hw_scale_response = real(ifft(sum(hw_num .* hw_xsf, 1) ./ (hw_den + params.scale_lambda) ));
                % find the maximum scale response
                recovered_hw_scale = find(hw_scale_response == max(hw_scale_response(:)), 1);
                % update the scale
                current_hw_ScaleFactor = current_hw_ScaleFactor.*(hw_scaleFactors(:,recovered_hw_scale))';
            end
            iter = iter + 1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % Extract sample and init projection matrix
    if seq.frame == 1
        % Extract image region for training sample
        sample_pos = round(pos);
        sample_scale = currentScaleFactor;
        
        xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
        % Do windowing of features
        xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
        % Compute the fourier series
        xlf = cellfun(@cfft2, xlw, 'uniformoutput', false);
        % Interpolate features to the continuous domain
        xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
        % New sample to be added
        xlf = compact_fourier_coeff(xlf);
        % Shift sample
        shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
        xlf_proj = shift_sample(xlf, shift_samp, kx, ky);
        
        clear xlw
    elseif params.learning_rate > 0
        % Use the sample that was used for detection
        sample_scale = sample_scale(scale_ind);
        xlf_proj = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:,scale_ind), xtf_proj, 'uniformoutput', false);
        
        % Shift the sample so that the target is centered
        shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
        xlf_proj = shift_sample(xlf_proj, shift_samp, kx, ky);
    end
    
    %%
    % Side samples
    if params.lambda2 ~=0
        if train_tracker
            for i = 1:4
                xl_side = extract_features_multi(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info, i);
                % Do windowing of features
                xlw_side = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_side, cos_window, 'uniformoutput', false);
                % Compute the fourier series
                xlf_side = cellfun(@cfft2, xlw_side, 'uniformoutput', false);
                % Interpolate features to the continuous domain
                xlf_side = interpolate_dft(xlf_side, interp1_fs, interp2_fs);
                % New sample to be added
                xlf_side = compact_fourier_coeff(xlf_side);
                % Shift sample
                shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
                xlf_proj_side{i} = shift_sample(xlf_side, shift_samp, kx, ky);
                clear xlw_side
            end
        end
    end
    
    
    
    %%
    xlf_proj_perm = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj, 'uniformoutput', false);
    if params.lambda2~=0
        if train_tracker
            xlf_proj_perm1 = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj_side{1}, 'uniformoutput', false);
            xlf_proj_perm2 = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj_side{2}, 'uniformoutput', false);
            xlf_proj_perm3 = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj_side{3}, 'uniformoutput', false);
            xlf_proj_perm4 = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj_side{4}, 'uniformoutput', false);
        end
    end
    
    % Do the traditional adding of a training sample and weight update
    % of C-COT
    [prior_weights, replace_ind] = update_prior_weights(prior_weights, gather(sample_weights), latest_ind, seq.frame, params);
    latest_ind = replace_ind;
    
    merged_sample_id = 0;
    new_sample_id = replace_ind;
    if params.use_gpu
        new_sample = xlf_proj;
    else
        new_sample = xlf_proj_perm;
    end
    
    % Insert the new training sample
    for k = 1:num_feature_blocks
        if merged_sample_id > 0
            samplesf{k}(merged_sample_id,:,:,:) = merged_sample{k};
        end
        if new_sample_id > 0
            samplesf{k}(new_sample_id,:,:,:) = new_sample{k};
        end
    end
    
    sample_weights = cast(prior_weights, 'like', params.data_type);
    
    if train_tracker
        if seq.frame == 1
            CG_state = [];
            
            CG_opts.maxit = params.init_CG_iter; % Number of initial iterations if projection matrix is not updated
            
            hf = cell(1,1,num_feature_blocks);
            
            % Initialize the filter with zeros
            for k = 1:num_feature_blocks
                hf{1,1,k} = zeros([filter_sz(k,1) (filter_sz(k,2)+1)/2 sample_dim(k)], 'like', params.data_type_complex);
            end
        else
            CG_opts.maxit = params.CG_iter;
        end
        
        % Do training
        % Do Conjugate gradient optimization of the filter
        if params.lambda2 ~=0
            [hf, res_norms, CG_state] = train_filter2(hf, samplesf, yf, reg_filter, sample_weights, params, CG_opts, CG_state, xlf_proj_perm1,xlf_proj_perm2,xlf_proj_perm3,xlf_proj_perm4);
        else
            [hf, res_norms, CG_state] = train_filter2(hf, samplesf, yf, reg_filter, sample_weights, params, CG_opts, CG_state);
        end
        
        % Reconstruct the full Fourier series
        hf_full = full_fourier_coeff(hf);
        
        frames_since_last_train = 0;
    else
        frames_since_last_train = frames_since_last_train+1;
    end
    
%     if params.visual_debug~=0
%         for i = 1:4
%             xl_side2 = extract_features_multi(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info, i);
%             % Do windowing of features
%             xlw_side2 = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_side2, cos_window, 'uniformoutput', false);
%             % Compute the fourier series
%             xlf_side2 = cellfun(@cfft2, xlw_side2, 'uniformoutput', false);
%             % Interpolate features to the continuous domain
%             xlf_side2 = interpolate_dft(xlf_side2, interp1_fs, interp2_fs);
%             % New sample to be added
%             xlf_side2 = compact_fourier_coeff(xlf_side2);
%             % Shift sample
%             shift_samp2 = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
%             xlf_proj_side2{i} = shift_sample(xlf_side2, shift_samp2, kx, ky);
%             clear xlw_side
%             AAA = cell2mat(xlf_proj_side2{i}([1:1]));
%             BBB = cell2mat(xlf_proj_side2{i}([2:2]));
%             
%             XXX = cell2mat(hf([1:1]));
%             YYY = cell2mat(hf([2:2]));
%             orth_f = [XXX(:);YYY(:)];
%             orth_x(:,i) = [AAA(:);BBB(:)];
%             orth(seq.frame,i) = rad2deg(subspace(orth_x(:,i),orth_f));
%         end
%     end
    
    
    if nScales > 0 && params.use_scale_filter
        if seq.frame == 1
            sf_den = [];
            sf_num = [];
        end
        [sf_den, sf_num] = scale_filter_update(im, pos, base_target_sz, currentScaleFactor, scaleFactors, scale_window, scale_model_sz, params, ysf, seq.frame, sf_den, sf_num);
    end
    if params.use_hw_scale_filter
        
        if seq.frame == 1
            hw_den = [];
            hw_num = [];
        end
        [hw_den, hw_num] = scale_hw_filter_update(im, pos, base_target_sz, current_hw_ScaleFactor, currentScaleFactor, hw_scaleFactors, hw_scale_window, scale_model_sz, params, hw_ysf, seq.frame, hw_den, hw_num);
    end
    
    % Update the target size (only used for computing output box)
    if params.use_hw_scale_filter
        target_sz = base_target_sz * currentScaleFactor.*current_hw_ScaleFactor;
    else
        target_sz = base_target_sz * currentScaleFactor;
    end
    
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % debug visualization
    if params.debug
        figure(20)
        %         set(gcf,'units','normalized','outerposition',[0 0 1 1]);
        subplot_cols = num_feature_blocks;
        subplot_rows = 3;%ceil(feature_dim/subplot_cols);
        for disp_layer = 1:num_feature_blocks
            subplot(subplot_rows,subplot_cols,disp_layer);
            imagesc(mean(abs(sample_fs(conj(hf_full{disp_layer}))), 3));
            colorbar;
            axis image;
            subplot(subplot_rows,subplot_cols,disp_layer+subplot_cols);
            imagesc(mean(abs(xl{disp_layer}), 3));
            colorbar;
            axis image;
            if seq.frame > 1
                subplot(subplot_rows,subplot_cols,disp_layer+2*subplot_cols);
                imagesc(fftshift(sample_fs(scores_fs_feat{disp_layer}(:,:,1,scale_ind))));
                colorbar;
                axis image;
            end
        end
        
        if train_tracker
            residuals_pcg = [residuals_pcg; res_norms];
            res_start_ind = max(1, length(residuals_pcg)-300);
            figure(99);plot(res_start_ind:length(residuals_pcg), residuals_pcg(res_start_ind:end));
            axis([res_start_ind, length(residuals_pcg), 0, min(max(residuals_pcg(res_start_ind:end)), 0.2)]);
        end
    end
    
    % visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame == 1  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
            %             set(fig_handle, 'Position', [100, 100, size(im,2), size(im,1)]);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            % Do visualization of the sampled confidence scores overlayed
%             resp_sz = round(img_support_sz*currentScaleFactor*scaleFactors(scale_ind));
%             xs = floor(det_sample_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
%             ys = floor(det_sample_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
        end
        
        drawnow
    end
end

% close(writer);
[~, results] = get_sequence_results(seq);
if params.visual_debug~=0
    results.orth = orth;
end

disp(['fps: ' num2str(results.fps)])

