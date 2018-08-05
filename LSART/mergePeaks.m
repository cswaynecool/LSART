function merged_peaks = mergePeaks(peaks, rad)

dis_mat = pdist2(peaks,peaks) + diag(inf*ones(size(peaks,1),1));
while min(dis_mat(:)) < rad && size(peaks,1) > 1
    [val idx] = min(dis_mat(:));
    [id1 id2] = ind2sub(size(dis_mat),idx);
    merged_peak = 0.5*(peaks(id1,:) + peaks(id2,:));
    peaks([id1 id2],:) = [];
    peaks = [peaks;merged_peak];
    dis_mat = pdist2(peaks,peaks) + diag(inf*ones(size(peaks,1),1));
end

merged_peaks = peaks;

end