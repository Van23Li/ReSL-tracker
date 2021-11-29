function [nScales, scale_step, scaleFactors, scale_window, scale_model_sz, params, ysf] = init_scale_filter_DSST(params)

% Initialize the scale filter parameters. Uses the DSST scale filter.

nScales = params.num_scales;

scale_step = params.scale_step;

scale_sigma = sqrt(params.num_scales) * params.scale_sigma_factor;

ss = (1:params.num_scales) - ceil(params.num_scales/2);

ys = exp(-0.5 * (ss.^2) / scale_sigma^2);

ysf = single(fft(ys));

if mod(params.num_scales,2) == 0
    scale_window = single(hann(params.num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(params.num_scales));
end

ss = 1:params.num_scales;

scaleFactors = params.scale_step.^(ceil(params.num_scales/2) - ss);

if params.scale_model_factor^2 * prod(params.init_sz) > params.scale_model_max_area
    params.scale_model_factor = sqrt(params.scale_model_max_area/prod(params.init_sz));
end

if prod(params.init_sz) > params.scale_model_max_area
    params.scale_model_factor = sqrt(params.scale_model_max_area/prod(params.init_sz));
end

scale_model_sz = floor(params.init_sz * params.scale_model_factor);
