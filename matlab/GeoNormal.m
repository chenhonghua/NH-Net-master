classdef GeoNormal < handle
    %GEONORMAL 
    
    properties
        % cluster
        cluster_model_;
        
        % parameter
        sigma_s_;
        sigma_r_;
        self_included_;
        rotate_feature_;
        map_size_;
    end
    
    methods (Static)
        function errors = compute_errors(N, Ng)
            errors = sum((N - Ng).^2);
            errors = acosd(1 - errors/2);
        end
    end
    
    methods
        function obj = GeoNormal(sigma_s, sigma_r, rotate_feature, self_included, ...
                map_size, pca_k, cluster_k, cluster_threshold)
            
            obj.rotate_feature_ = logical(rotate_feature);
            obj.self_included_ = logical(self_included);
            obj.sigma_s_ = sigma_s;
            obj.sigma_r_ = sigma_r;
            obj.map_size_ = map_size;
            
            % cluster model
            obj.cluster_model_ = ClusterModel(pca_k, cluster_k, cluster_threshold);
            
        end
        
        
        function obj = run(obj, path_noisy, path_original, path_result, path_train, Ks)

            files = dir([path_noisy, '*.obj']);
            nfiles = length(files);
            pnums = zeros(1, nfiles);
            mean_errors = zeros(1, nfiles);
            
            % normal estimation
            Nfs = cell(1, nfiles);
            Ngs = cell(1, nfiles);
            Rots = cell(1, nfiles);
            HMPs = cell(1, nfiles);
            
            min_k = min(Ks);
            knn_map = ceil(1.5*min_k);
            knn_BNF = ceil(1.5*min_k);
            
            parfor i = 1:nfiles
                
                [V, ~] = read_mesh([path_noisy, files(i).name]);
                pts = V';
                pnums(i) = length(pts);
                
                % read corresponding ground truth mesh model
                filename_original = [path_original, files(i).name(1:end-7), '.obj'];
                [Vm, Fm] = read_mesh(filename_original);
                
                % estimate normal
                [normals, ~] = NormalEstimate_fps_MS_Ew(pts, Ks, 150);
                FN = compute_face_normal(Vm, Fm);
                [N_reori, gt_normals] = normal_reorientation_mesh(normals', FN, Fm);
                write_xyz([path_result, files(i).name(1:end-4), '.xyz'], pts', N_reori);
                errors = obj.compute_errors(N_reori, gt_normals);
                mean_errors(i) = mean(errors);
                
                % Feature
                [feature, rot_, gt_normals] = calc_feature(obj.sigma_s_, obj.sigma_r_, pts, ...
                    knn_BNF, N_reori', obj.rotate_feature_, obj.self_included_, gt_normals);

                % Height Map 
                hmp_ = compute_HeightMap1(pts, feature', rot_, obj.map_size_, knn_map);
                %hmp_ = permute(hmp_, [4,3,1,2]);
                
                % collect data
                Nfs{i} = feature';
                Ngs{i} = gt_normals';
                Rots{i} = rot_;
                HMPs{i} = hmp_;
            end

            % collect data
            nid = 1;
            for i = 1:length(files)
                id = nid : nid+pnums(i)-1;
                HMP(id,:,:,:) = HMPs{i};
                Nf(id,:) = Nfs{i};
                Ng(id,:) = Ngs{i};
                Rot(:,:,id) = Rots{i};
                nid = nid+pnums(i);
            end
            
            % compute cluster
            [idx_cluster, ~, Nf] = obj.cluster_model_.init_cluster(Nf', Ng');
            
            % data saved to file
            writeNPY(single(HMP), [path_train, 'HMP.npy']);
            writeNPY(single(Nf), [path_train, 'Nf.npy']);
            writeNPY(single(Ng), [path_train, 'Ng.npy']);
            % to row-major
            Rot = permute(Rot, [3,1,2]); % rotate: rot*n, re-: rot'*n
            writeNPY(single(Rot), [path_train, 'Rot.npy']);

            % Save Info
            writeNPY(idx_cluster, [path_train, 'idx_cluster.npy']);
            fid = fopen([path_train, 'filenames.txt'], 'w');
            for i = 1:length(files)
                fprintf(fid, '%s %d\n', files(i).name(1:end-4), pnums(i));
            end
            fclose(fid);
            
            % Save Error
            fid = fopen([path_result, 'ErrorInfo.txt'], 'w');
            fprintf(fid, '%d  %d  %f\n', nfiles, sum(pnums), sum(pnums .* mean_errors) / sum(pnums));

            for i = 1:length(files)
                fprintf(fid, '%s  %d  %f\n', files(i).name, pnums(i), mean_errors(i));
            end
            fclose(fid);
        end
        
        
        function obj = test(obj, test_noisy, test_result, Ks)
            
            files = dir([test_noisy, '*.off']);
            nfiles = length(files);
            pnums = zeros(1, nfiles);
            
            % normal estimation
            Nfs = cell(1, nfiles);
            Rots = cell(1, nfiles);
            HMPs = cell(1, nfiles);
            
            min_k = min(Ks);
            knn_map = ceil(1.5*min_k);
            knn_BNF = ceil(1.5*min_k);
            parfor i = 1:nfiles
                
                pts = read_off([test_noisy, files(i).name]);
                pnums(i) = length(pts);
                
                % estimate normal
                [normals, ~] = NormalEstimate_fps_MS_Ew(pts, Ks, 150);
                write_xyz([test_result, files(i).name(1:end-4), '.xyz'], pts', normals');
                
                % Feature
                [feature, rot_] = calc_feature(obj.sigma_s_, obj.sigma_r_, pts, ...
                    knn_BNF, normals', obj.rotate_feature_, obj.self_included_);

                % Height Map 
                hmp_ = compute_HeightMap1(pts, feature', rot_, obj.map_size_, knn_map);
                
                % collect data
                Nfs{i} = feature';
                Rots{i} = rot_;
                HMPs{i} = hmp_;
            end

            % collect data
            nid = 1;
            for i = 1:length(files)
                id = nid : nid+pnums(i)-1;
                HMP(id,:,:,:) = HMPs{i};
                Nf(id,:) = Nfs{i};
                Rot(:,:,id) = Rots{i};
                nid = nid+pnums(i);
            end
            
            % compute cluster
            [idx_cluster, ~, Nf] = obj.cluster_model_.compute_cluster(Nf');
            
            % data saved to file
            writeNPY(single(HMP), [test_result, 'HMP.npy']);
            writeNPY(single(Nf), [test_result, 'Nf.npy']);
            % to row-major
            Rot = permute(Rot, [3,1,2]);
            writeNPY(single(Rot), [test_result, 'Rot.npy']);

            % Save Info
            writeNPY(idx_cluster, [test_result, 'idx_cluster.npy']);
            fid = fopen([test_result, 'filenames.txt'], 'w');
            for i = 1:length(files)
                fprintf(fid, '%s %d\n', files(i).name(1:end-4), pnums(i));
            end
            fclose(fid);

        end
        
        
        function obj = test_from_normal(obj, test_result, Ks)
            
            files = dir([test_result, '*.xyz']);
            nfiles = length(files);
            pnums = zeros(1, nfiles);
            
            % normal estimation
            min_k = min(Ks);
            knn_map = ceil(1.5*min_k);
            knn_BNF = ceil(1.5*min_k);
            
            nid = 1;
            for i = 1:nfiles
                
                [V, N] = read_xyz([test_result, files(i).name]);
                pts = V';
                pnums(i) = length(pts);
                
                % Feature
                [feature, rot_] = calc_feature(obj.sigma_s_, obj.sigma_r_, pts, ...
                    knn_BNF, N, obj.rotate_feature_, obj.self_included_);

                % Height Map 
                hmp_ = compute_HeightMap1(pts, feature', rot_, obj.map_size_, knn_map);
                
                % collect data
                id = nid : nid+pnums(i)-1;
                HMP(id,:,:,:) = hmp_;
                Nf(id,:) = feature';
                Rot(:,:,id) = rot_;
                nid = nid+pnums(i);
            end

            
            % compute cluster
            [idx_cluster, ~, Nf] = obj.cluster_model_.compute_cluster(Nf');
            
            % data saved to file
            writeNPY(single(HMP), [test_result, 'HMP.npy']);
            writeNPY(single(Nf), [test_result, 'Nf.npy']);
            % to row-major
            Rot = permute(Rot, [3,1,2]);
            writeNPY(single(Rot), [test_result, 'Rot.npy']);

            % Save Info
            writeNPY(idx_cluster, [test_result, 'idx_cluster.npy']);
            fid = fopen([test_result, 'filenames.txt'], 'w');
            for i = 1:length(files)
                fprintf(fid, '%s %d\n', files(i).name(1:end-4), pnums(i));
            end
            fclose(fid);

        end
        
    end
end

