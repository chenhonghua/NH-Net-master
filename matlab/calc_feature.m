function [feature, Rot, gt_normals] = calc_feature(sigma_s, sigma_r, pts, ...
    knn_BNF, normals, rotate_feature, self_included, gt_normals)
%CAL_FEATURE 

if size(pts, 1) == 3, pts = pts'; end
if size(normals, 1) == 3, normals = normals'; end

% Filtering
[feature, Rot] = NormalFilter_MS_Nor(sigma_s, sigma_r, pts, knn_BNF, normals, ...
    rotate_feature, self_included);

% to z-axis +
for i = 1:length(feature)
    if feature(3,i)<0
        feature(:,i) = -feature(:,i);
        Rot(:,:,i) = -Rot(:,:,i);
    end
end

% Ng rotate
if exist('gt_normals', 'var')
    if rotate_feature
        for i = 1:length(feature)
            gt_normals(:,i) = Rot(:,:,i) * gt_normals(:,i);
        end
    end
else
    gt_normals = [];
end

end

