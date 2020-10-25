function [ feature, Rot ] = NormalFilter_MS_Nor( sigma_s, sigma_r, pts, ...
                k_knn, normals, rotate_feature, self_included)
%MULTISCALE_BNF 

npts = size(pts, 1);
ns = length(sigma_s);
nr = length(sigma_r);
% sigma_r(end) = 1e9;
n_input = ns*nr*3;
if self_included, n_input = n_input+3; end
feature = zeros(n_input, npts);
Rot = zeros(3, 3, npts);

%% bilateral normal filtering
kdtree = kdtree_build(pts);

for i = 1:npts
    R = zeros(3,3);
    len = compute_avg_length(kdtree, pts, i, ceil(0.1*k_knn));
%     len = compute_max_length(kdtree, pts, i, ceil(0.1*k_knn));

    knn = kdtree_k_nearest_neighbors(kdtree, pts(i,:), k_knn);
    pts_knn = pts(knn,:);
    
    dis = pts_knn - repmat(pts(i,:), k_knn, 1);
    dis = sqrt(sum(dis.^2, 2));
    
    normals_knn = normals(knn,:);
    temp = dot(normals_knn, repmat(normals(i,:), k_knn, 1), 2);
    normals_knn(temp<0, :) = -normals_knn(temp<0, :);
    ndis = normals_knn - repmat(normals(i,:), k_knn, 1);
    ndis = sqrt(sum(ndis.^2, 2));

    nid = 1;
    if self_included
        feature((nid*3-2):(nid*3), i) = normals(i,:)';
        nid = nid+1;
        R = R + normals(i,:)'*normals(i,:);
    end
    for ss = 1:ns
        for rr = 1:nr
            s = sigma_s(ss) * len;
            r = sigma_r(rr);
            
            E = exp(-(dis.^2)./(2*s*s)) .* exp(-(ndis.^2)./(2*r*r));
            n_sum = sum(E .* normals_knn, 1);
            n_sum = (n_sum/norm(n_sum))';
%             n_sum = single(n_sum);

            feature((nid*3-2):(nid*3),i) = n_sum;
            R = R+n_sum*n_sum';
            nid = nid+1;
        end
    end
    if rotate_feature
        [V, D] = eig(R);
        [~,ind] = sort(diag(D));
        Vs = V(:,ind);
        Rot(:,:,i) = Vs';
        for j = 1:n_input/3
            feature((j*3-2):(j*3), i) = Vs' * feature((j*3-2):(j*3), i);
        end
    end
end

kdtree_delete(kdtree);
end


function [len] = compute_max_length(kdtree, pts, pid, k_knn)
%COMPUTE_MAX_LEN 

knn = kdtree_k_nearest_neighbors(kdtree, pts(pid,:), k_knn+1);
pts_knn = pts(knn,:);
dis = pts_knn - repmat(pts(pid,:), k_knn+1, 1);
dis = sqrt(sum(dis.^2, 2));
len = max(dis);

end

function [len] = compute_avg_length(kdtree, pts, pid, k_knn)
%COMPUTE_MAX_LEN 

knn = kdtree_k_nearest_neighbors(kdtree, pts(pid,:), k_knn+1);
pts_knn = pts(knn,:);
dis = pts_knn - repmat(pts(pid,:), k_knn+1, 1);
dis = sqrt(sum(dis.^2, 2));
len = sum(dis)/k_knn;

end
