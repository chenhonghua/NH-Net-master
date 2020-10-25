function [N2, gt_normals] = normal_reorientation(N1, mesh_normals)
% reorient using mesh normals (cell(1,x))

if size(N1,1) ~= 3
    N1 = N1';
end
npts = length(N1);
N2 = zeros(3, npts);
gt_normals = zeros(3, npts);
for i = 1:npts
    nsize = size(mesh_normals{i}, 2);
    gt = mesh_normals{i};
    ni = N1(:,i);
    max_theta = 0;
    for j = 1:nsize
        c = norm(gt(:,j) - ni);
        theta = abs(1-c*c/2);
        if theta>max_theta
            max_theta = theta;
            gt_normals(:,i) = gt(:,j);
        end
    end
    if gt_normals(:,i)' * ni < 0
        N2(:,i) = -N1(:,i);
    else
        N2(:,i) = N1(:,i);
    end
end

end

