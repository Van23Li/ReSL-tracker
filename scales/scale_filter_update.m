function [sf_den, sf_num] = scale_filter_update(im, pos, base_target_sz, currentScaleFactor, scaleFactors, scale_window, scale_model_sz, params, ysf, frame, sf_den, sf_num)

% Update the scale filter.

    xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
    xsf = fft(xs,[],2);%每一行做fft
    new_sf_num = bsxfun(@times, ysf, conj(xsf));%算互相关
    new_sf_den = sum(xsf .* conj(xsf), 1);% 自相关，每一列都加起来，是1*33维的一个向量
    
        if frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - params.learning_rate_scale) * sf_den + params.learning_rate_scale * new_sf_den;
        sf_num = (1 - params.learning_rate_scale) * sf_num + params.learning_rate_scale * new_sf_num;
        end