function hf_out = lhs_operation2(hf, samplesf,samplesf1,samplesf2,samplesf3,samplesf4, reg_filter, sample_weights, params)

% This is the left-hand-side operation in Conjugate Gradient

% Get sizes
num_features = length(hf);
filter_sz = zeros(num_features,2);
for k = 1:num_features
    filter_sz(k,:) = [size(hf{k},1), size(hf{k},2)];
end
[~, k1] = max(filter_sz(:,1));  % Index for the feature block with the largest spatial size
block_inds = 1:num_features;
block_inds(k1) = [];
output_sz = [size(hf{k1},1), 2*size(hf{k1},2)-1];

% Compute the operation corresponding to the data term in the optimization
% (blockwise matrix multiplications)
%implements: A' diag(sample_weights) A f

%%
% sum over all features and feature blocks
sh0 = mtimesx(samplesf{k1}, permute(hf{k1}, [3 4 1 2]), 'speed');    % assumes the feature with the highest resolution is first
pad_sz = cell(1,1,num_features);
for k = block_inds
    pad_sz{k} = (output_sz - [size(hf{k},1), 2*size(hf{k},2)-1]) / 2;
    
    sh0(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) = ...
        sh0(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) + mtimesx(samplesf{k}, permute(hf{k}, [3 4 1 2]), 'speed');
end

% weight all the samples
sh0 = bsxfun(@times,sample_weights,sh0);

% multiply with the transpose
hf_out0 = cell(1,1,num_features);
hf_out0{k1} = permute(conj(mtimesx(sh0, 'C', samplesf{k1}, 'speed')), [3 4 2 1]);
for k = block_inds
    hf_out0{k} = permute(conj(mtimesx(sh0(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end), 'C', samplesf{k}, 'speed')), [3 4 2 1]);
end
%%
% sum over all features and feature blocks
sh1 = mtimesx(samplesf1{k1}, permute(hf{k1}, [3 4 1 2]), 'speed');    % assumes the feature with the highest resolution is first
% pad_sz = cell(1,1,num_features);
for k = block_inds
%     pad_sz{k} = (output_sz - [size(hf{k},1), 2*size(hf{k},2)-1]) / 2;
    
    sh1(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) = ...
        sh1(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) + mtimesx(samplesf1{k}, permute(hf{k}, [3 4 1 2]), 'speed');
end

% weight all the samples
% sh1 = bsxfun(@times,sample_weights,sh1);

% multiply with the transpose
hf_out1 = cell(1,1,num_features);
hf_out1{k1} = permute(conj(mtimesx(sh1, 'C', samplesf1{k1}, 'speed')), [3 4 2 1]);
for k = block_inds
    hf_out1{k} = permute(conj(mtimesx(sh1(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end), 'C', samplesf1{k}, 'speed')), [3 4 2 1]);
end
%%
% sum over all features and feature blocks
sh2 = mtimesx(samplesf2{k1}, permute(hf{k1}, [3 4 1 2]), 'speed');    % assumes the feature with the highest resolution is first
% pad_sz = cell(1,1,num_features);
for k = block_inds
%     pad_sz{k} = (output_sz - [size(hf{k},1), 2*size(hf{k},2)-1]) / 2;
    
    sh2(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) = ...
        sh2(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) + mtimesx(samplesf2{k}, permute(hf{k}, [3 4 1 2]), 'speed');
end

% weight all the samples
% sh2 = bsxfun(@times,sample_weights,sh2);

% multiply with the transpose
hf_out2 = cell(1,1,num_features);
hf_out2{k1} = permute(conj(mtimesx(sh2, 'C', samplesf2{k1}, 'speed')), [3 4 2 1]);
for k = block_inds
    hf_out2{k} = permute(conj(mtimesx(sh2(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end), 'C', samplesf2{k}, 'speed')), [3 4 2 1]);
end
%%
% sum over all features and feature blocks
sh3 = mtimesx(samplesf3{k1}, permute(hf{k1}, [3 4 1 2]), 'speed');    % assumes the feature with the highest resolution is first
% pad_sz = cell(1,1,num_features);
for k = block_inds
%     pad_sz{k} = (output_sz - [size(hf{k},1), 2*size(hf{k},2)-1]) / 2;
    
    sh3(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) = ...
        sh3(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) + mtimesx(samplesf3{k}, permute(hf{k}, [3 4 1 2]), 'speed');
end

% weight all the samples
% sh3 = bsxfun(@times,sample_weights,sh3);

% multiply with the transpose
hf_out3 = cell(1,1,num_features);
hf_out3{k1} = permute(conj(mtimesx(sh3, 'C', samplesf3{k1}, 'speed')), [3 4 2 1]);
for k = block_inds
    hf_out3{k} = permute(conj(mtimesx(sh3(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end), 'C', samplesf3{k}, 'speed')), [3 4 2 1]);
end
%%
% sum over all features and feature blocks
sh4 = mtimesx(samplesf4{k1}, permute(hf{k1}, [3 4 1 2]), 'speed');    % assumes the feature with the highest resolution is first
% pad_sz = cell(1,1,num_features);
for k = block_inds
%     pad_sz{k} = (output_sz - [size(hf{k},1), 2*size(hf{k},2)-1]) / 2;
    
    sh4(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) = ...
        sh4(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end) + mtimesx(samplesf4{k}, permute(hf{k}, [3 4 1 2]), 'speed');
end

% weight all the samples
% sh4 = bsxfun(@times,sample_weights,sh4);

% multiply with the transpose
hf_out4 = cell(1,1,num_features);
hf_out4{k1} = permute(conj(mtimesx(sh4, 'C', samplesf4{k1}, 'speed')), [3 4 2 1]);
for k = block_inds
    hf_out4{k} = permute(conj(mtimesx(sh4(:,1,1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end), 'C', samplesf4{k}, 'speed')), [3 4 2 1]);
end
%%
% compute the operation corresponding to the regularization term (convolve
% each feature dimension with the DFT of w, and the tramsposed operation)
% add the regularization part
% hf_conv = cell(1,1,num_features);

    hf_out = cell(1,1,num_features);
    
for k = 1:num_features
    reg_pad = min(size(reg_filter{k},2)-1, size(hf{k},2)-1);
    
    % add part needed for convolution
    hf_conv = cat(2, hf{k}, conj(rot90(hf{k}(:, end-reg_pad:end-1, :), 2)));
    
    % do first convolution
    hf_conv = convn(hf_conv, reg_filter{k});
    
    % do final convolution and put toghether result
%     hf_out{k} = hf_out{k} + convn(hf_conv(:,1:end-reg_pad,:), reg_filter{k}, 'valid');
    hf_out{k} = hf_out0{k} + params.lambda2 .* (hf_out1{k} + hf_out2{k} + hf_out3{k} + hf_out4{k}) + convn(hf_conv(:,1:end-reg_pad,:), reg_filter{k}, 'valid');
end

end