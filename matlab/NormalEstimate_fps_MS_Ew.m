function [normVectors, range] = NormalEstimate_fps_MS_Ew(pts, Ks, rand_num)
% Multi-scale fitting patch selection
if size(pts, 1) == 3
    pts = pts';
end
npts = size(pts, 1);

kdtree = kdtree_build(pts);
max_k = max(Ks);
min_k = min(Ks);
range = zeros(npts, 1);
g_normals = zeros(npts, 3);

%% Initialize PCA normals
% k_bw = min_k; 
% 
% k_pca = min_k;
% k_pca = ceil(0.5*max_k); 

k_bw = ceil(0.7*min_k);
k_pca = ceil(0.7*min_k);

k1 = ceil(0.5*k_pca);
TP.sigma_threshold = 0.05;
TP.debug_taproot = 0;
[sigms , normVectors , ~ , ~] = compute_points_sigms_normals_two(pts, k_pca, kdtree, k1);

feature_threshold = feature_threshold_selection(sigms,TP);

init_feat = find(sigms > feature_threshold);
feature_sigms = sigms(init_feat);
[~, id_sigms] = sort(feature_sigms);
id_feature = init_feat(id_sigms);
nfeatures = length(id_feature);

bandwidth = compute_bandwidth_points(pts, kdtree, k_bw, 1.5*max_k); % 2*k_bw

%%
voting = zeros(1,npts);
for kk=1:length(Ks)
    k_knn = Ks(1,kk);
        
    for i = 1:npts
        knn = kdtree_k_nearest_neighbors(kdtree, pts(i,:), k_knn);
        pts_knn = pts(knn,:);
       % rand
        knn_feature = intersect(knn, id_feature);
        if isempty(knn_feature), continue; end
        max_sort = 0;
        inner_threshold = 2*bandwidth(1,i);
        tempx = ones(1 , 4) ;
        tempy = ones(k_knn , 1);
        for j = 1 : rand_num
            ran_id = [ 1 , randperm(k_knn - 1 , 3) + 1 ]; 
            points_3 = pts_knn(ran_id , :) ;

            mp = (tempx*points_3)./4;
            points_center = pts_knn - tempy * mp;
            
            tmp = points_center(ran_id , :); 
            C = tmp'*tmp./size(points_3 , 1); 
            [V,~] = eig(C); 
            fix_normal = V(:,1); 

            dis = (points_center*fix_normal).^2; 
            if dis(1) > inner_threshold^2
                continue;
            end

            dis = exp(-dis./min( inner_threshold.^2 ) ); 

            cur_sort = sum(dis);

            if cur_sort > max_sort
                max_sort = cur_sort;
                n_patch = fix_normal ;
                mp_patch = mp;
            end

        end

        
       % consistency measurement of selected patch
        d = (pts_knn - repmat(mp_patch, k_knn, 1)) * n_patch;
        d = d.^2;
        for j=1:length(knn_feature)
            fid = knn_feature(j);
            bd = bandwidth(1,fid)+1e-9;

            temp = d./((2*bd)^2);
            temp = exp(-temp);
            E = sum(temp);
            E = (E/k_knn) * exp(-((1 - k_knn/max_k)/3).^2); % Patch Size
            
            dis = abs((pts(fid,:) - pts(i,:))*n_patch);
            E = E * exp(-(dis/(2*bd))^2);  % E(patch)*w(dis_feaPoint)
            if E>voting(1,fid)
                voting(1,fid) = E;
                g_normals(fid,:) = n_patch';
                range(fid,1) = k_knn;
            end
        end
    end
end

normVectors(id_feature,:) = g_normals(id_feature,:);

kdtree_delete(kdtree);
       
end

