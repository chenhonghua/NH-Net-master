function [N1, gt_normals] = normal_reorientation_mesh(N, FN, F)

npts = length(N);
voting = zeros(1,npts);
gt_normals = zeros(3,npts);
nfaces = length(F);

% ground truth normal
for f = 1:nfaces
    for i = 1:3
        pid = F(i,f);
        t = abs(FN(:,f)' * N(:,pid));
        if t > voting(1,pid)
            voting(1,pid) = t;
            gt_normals(:,pid) = FN(:,f);
        end
    end
end

% reorientation
N1 = N;
for i = 1:npts
    if N1(:,i)' * gt_normals(:,i) < 0
        N1(:,i) = -N1(:,i);
    end
end

end

