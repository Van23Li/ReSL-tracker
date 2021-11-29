function [hw_nScales, hw_scale_step, hw_scaleFactors, hw_scale_window, params, hw_ysf] = init_hw_scale_filter_DSST(params)

% Initialize the scale filter parameters. Uses the DSST scale filter.

hw_nScales = params.num_hw_scales;

hw_scale_step = params.hw_scale_step;

hw_scale_sigma = sqrt(params.num_hw_scales) * params.scale_sigma_factor;

        hw_ss = (1:params.num_hw_scales) - ceil(params.num_hw_scales/2);
        
        hw_ys = exp(-0.5 * (hw_ss.^2) / hw_scale_sigma^2);
        
        hw_ysf = single(fft(hw_ys));
        
        if mod(params.num_hw_scales,2) == 0
            hw_scale_window = single(hann(params.num_hw_scales+1));
            hw_scale_window = hw_scale_window(2:end);
        else
            hw_scale_window = single(hann(params.num_hw_scales));
        end
        
        hw_ss = 1:params.num_hw_scales;
        
        hw_scaleFactors = params.hw_scale_step.^((ceil(params.num_hw_scales/2) - hw_ss));
        hw_scaleFactors = [hw_scaleFactors;2-hw_scaleFactors];