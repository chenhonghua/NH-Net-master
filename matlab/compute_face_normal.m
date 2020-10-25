function [Nf, areas] = compute_face_normal(V, F)


	Nf = cross( V(:,F(2,:))-V(:,F(1,:)), V(:,F(3,:))-V(:,F(1,:)));
	d = sqrt( sum(Nf.^2) ); 

	areas = 0.5*d;
	d(d<eps) = eps;
	Nf =  bsxfun(@rdivide, Nf, d);

end