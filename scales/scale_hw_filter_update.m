function [hw_den, hw_num] = scale_hw_filter_update_DSST(im, pos, base_target_sz, current_hw_ScaleFactor, currentScaleFactor, hw_scaleFactors, hw_scale_window, scale_model_sz, params, hw_ysf, frame, hw_den, hw_num)

% Update the scale filter.

    hw_xs=get_scale_hw_sample(im, pos, base_target_sz.*current_hw_ScaleFactor, currentScaleFactor,hw_scaleFactors, hw_scale_window, scale_model_sz);
    hw_xsf=fft(hw_xs,[],2);
    new_hw_num = bsxfun(@times, hw_ysf, conj(hw_xsf)); %算互相关
    new_hw_den = sum(hw_xsf .* conj(hw_xsf), 1);% 自相关，每一列都加起来，是1*33维的一个向量
    
        if frame == 1
        hw_den = new_hw_den;
        hw_num = new_hw_num;
        else
        hw_den = (1 - params.learning_rate_hw_scale) * hw_den + params.learning_rate_hw_scale * new_hw_den;
        hw_num = (1 - params.learning_rate_hw_scale) * hw_num + params.learning_rate_hw_scale * new_hw_num;
    end