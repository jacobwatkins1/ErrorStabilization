%
% N is the dimension of the large-space target Hamiltonian, 
% Ht = H0 + target_coupling*H1
%
% order is the number of basis vectors we use in the eigenvector
% continuation (EC) calculation
%
% EC_step_coupling control the step increments we use to define the EC
% vectors for the variational calculation, |v_i>
%
% nmat_exact = < v_i | v_j >
% hmat_exact = < v_i | Ht | v_j >
%
% errsize_nmat is the standard deviation of the Gaussian errors in nmat;
% for simplicity is the same for all elements
%
% errsize_hmat is the standard deviation of the Gaussian errors in hmat;
% for simplicity is the same for all elements
%
% nmat_start and hmat_start have these Gaussian errors added to nmat_exact
% and hmat_exact respectively
%
% e(k) is the eigenvalue we get in the variational calculation at order k
% 
% convergence_ratio is the prior probability constraint that 
% |e(3) - e(2)| < convergence_ratio*|e(2) - e(1)|
% |e(4) - e(3)| < convergence_ratio*|e(3) - e(2)| 
% ...
%
% lowest_order_ratio is the lowest order in which we apply the convergence
% ratio constraint; the default value is 3
%
% nmat_delta is the width parameter for our logistic function which we use 
% for importance sampling of matrices nmat 
%
% logistic function is applied to mineig, the lowest eigenvalue of nmat,
% which suppresses the contribution of unphysical matrices nmat where mineig is 
% negative
%
% ntrials_nmat is the number of samples we try for nmat
%
% ntrials_hmat is minimum number of samples we try for hmat; the actual
% number of samples for hmat is controlled by the Markov chain weight for 
% the corresponding nmat matrix
%
% we use Metropolis sampling for nmat, but we use simple Gaussian sampling 
% for hmat
%
% although inefficient, we use simple Gaussian sampling for hmat because 
% the selection probability needs to have the correct absolute normalization
%
% E0 is the true ground state energy
%
% e_exact gives the order-by-order exact variational results for E0 using
% nmat_exact, hmat_exact
%
% e_start gives the order-by-order starting variational results for E0 using
% nmat_start, hmat_start
%
% e_list gives a list of the order-by-order variational results we obtain by 
% sampling over nmat and hmat configurations
%
% the eigenvalue results are displayed as 
% [e_exact; e_start; mean(e_list,1); std(e_list,1)]
%
% overlap_start gives the order-by-order inner products between the 
% variational results obtained using (nmat_start, hmat_start) and 
% the variational results obtained using (nmat_exact, hmat_exact) at the 
% same order
%
% overlap_list gives a list of the order-by-order inner products between the 
% variational results obtained by sampling (nmat, hmat) and 
% the variational results obtained using (nmat_exact, hmat_exact) at the 
% same order
%
% the overlap results are displayed as
% [ones(1,order); overlap_start; mean(overlap_list,1); std(overlap_list,1)]
%
% max_order_overlap_exact gives the order-by-order inner products between the 
% variational results obtained using (nmat_exact, hmat_exact) and 
% the variational result obtained using (nmat_exact, hmat_exact) at 
% the maximum order
%
% max_order_overlap_start gives the order-by-order inner products between the 
% variational results obtained using (nmat_start, hmat_start) and 
% the variational result obtained using (nmat_exact, hmat_exact) at 
% the maximum order
%
% max_order_overlap_list gives a list of the order-by-order inner products 
% between the variational results obtained by sampling (nmat, hmat) and 
% the variational result obtained using (nmat_exact, hmat_exact) at 
% the maximum order
%
% the max order results are displayed as
% [max_order_overlap_exact; max_order_overlap_start; ...
% mean(max_order_overlap_list,1); std(overlap_list,1)] 

clear all;

format long

N = 100;
order = 5;
lowest_order_ratio = 3;
convergence_ratio = 0.6;
EC_step_coupling = 0.1;
target_coupling = 1.0;
errsize_nmat = 0.001;
errsize_hmat = 0.01;
ntrials_nmat = 1000000;
ntrials_hmat = 10000;
metro_step = 0.1;
nmat_delta = 0.0001;

rng(3);

H0 = 2*rand(N,N)-1;
H1 = 2*rand(N,N)-1;
H0 = (H0+H0')/2.0;
H1 = (H1+H1')/2.0;

rng(13);

for nc = 1:order
    [vv,dd] = eig(H0 + nc*EC_step_coupling*H1);
    v(:,nc) = vv(:,1);
end

Ht = H0 + target_coupling*H1;
E0 = min(eig(Ht))

nmat_exact = v'*v
hmat_exact = v'*(Ht*v)

nmat_err = randn(order,order);
nmat_err = (nmat_err + nmat_err')/2*sqrt(2);
nmat_start = nmat_exact + errsize_nmat*nmat_err;

hmat_err = randn(order,order);
hmat_err = (hmat_err + hmat_err')/2*sqrt(2);
hmat_start = hmat_exact + errsize_hmat*hmat_err;

vsmall_exact = zeros(order,order);
for k = 1:order
    [vtemp,dtemp] = eig(nmat_exact(1:k,1:k)\hmat_exact(1:k,1:k));
    [notused,sorted] = sort(diag(dtemp));
    vsmall_exact(1:k,k) = vtemp(1:k,sorted(1));
end

for k = 1:order
    e_exact(k) = min(eig(nmat_exact(1:k,1:k)\hmat_exact(1:k,1:k)));
    e_start(k) = min(eig(nmat_start(1:k,1:k)\hmat_start(1:k,1:k)));    
end
e_exact
e_start

for k = 1:order
    [vtemp,dtemp] = eig(nmat_start(1:k,1:k)\hmat_start(1:k,1:k));
    [notused,sorted] = sort(diag(dtemp));
    vsmall_start(1:k,k) = vtemp(1:k,sorted(1));
    overlap_start(1,k) = ...
        abs(vsmall_exact(1:k,k)'*nmat_exact(1:k,1:k)*vsmall_start(1:k,k)) ...
        /sqrt(abs(vsmall_exact(1:k,k)'*nmat_exact(1:k,1:k)* ...
        vsmall_exact(1:k,k) ...
        *abs(vsmall_start(1:k,k)'*nmat_exact(1:k,1:k)*vsmall_start(1:k,k))));
    max_order_overlap_start(1,k) = ...
        abs(vsmall_exact(1:order,order)'*nmat_exact(1:order,1:k)*vsmall_start(1:k,k)) ...
        /sqrt(abs(vsmall_exact(1:order,order)'*nmat_exact(1:order,1:order)* ...
        vsmall_exact(1:order,order) ...
        *abs(vsmall_start(1:k,k)'*nmat_exact(1:k,1:k)*vsmall_start(1:k,k))));
    max_order_overlap_exact(1,k) = ...
        abs(vsmall_exact(1:order,order)'*nmat_exact(1:order,1:k)*vsmall_exact(1:k,k)) ...
        /sqrt(abs(vsmall_exact(1:order,order)'*nmat_exact(1:order,1:order)* ...
        vsmall_exact(1:order,order) ...
        *abs(vsmall_exact(1:k,k)'*nmat_exact(1:k,1:k)*vsmall_exact(1:k,k))));    
end
overlap_start(1,:)
max_order_overlap_start(1,:)
max_order_overlap_exact(1,:)

nmat = nmat_start;
mineig = min(eig(nmat));
guide = 1/(exp(-mineig/nmat_delta)+1);
gauss = exp(-sum(sum((nmat-nmat_start).^2))/(2*errsize_nmat^2));

e_list = zeros(1,order);
accept_nmat = 0;
accept_hmat = 0;

for ii = 1:ntrials_nmat
    
    nmat_eps = metro_step*errsize_nmat*randn(order,order);
    nmat_eps = (nmat_eps + nmat_eps')/2*sqrt(2);    
    nmat_new = nmat + nmat_eps;
    mineig_new = min(eig(nmat_new));
    guide_new = 1/(exp(-mineig_new/nmat_delta)+1);   
    gauss_new = exp(-sum(sum((nmat_new-nmat_start).^2))/(2*errsize_nmat^2));
    
    if (rand < (guide_new*gauss_new)/(guide*gauss))             
        nmat = nmat_new;
        guide = guide_new;
        gauss = gauss_new;        
        accept_nmat = accept_nmat + 1;
        mineig = mineig_new;         
    end    
    
    if (mineig > 0)

        fraction = ntrials_hmat/guide - floor(ntrials_hmat/guide);
        if (rand < fraction)
            iterations_hmat = floor(ntrials_hmat/guide) + 1;
        else
            iterations_hmat = floor(ntrials_hmat/guide);
        end
        for jj = 1:iterations_hmat
            hmat_err = randn(order,order);
            hmat_err = (hmat_err + hmat_err')/2*sqrt(2);
            hmat = hmat_start + errsize_hmat*hmat_err;
            for k = 1:order
                e(k) = min(eig(nmat(1:k,1:k)\hmat(1:k,1:k)));
            end
            concave = 1;
            for k = lowest_order_ratio:order
                if (abs(e(k-1)-e(k)) > convergence_ratio*abs(e(k-2)-e(k-1)))
                    concave = 0;
                end
            end
            if (concave == 1)
                accept_hmat = accept_hmat + 1;
                e_list(accept_hmat,:) = e; 
                if (size(e_list,1) > 1)
                    [e_exact; e_start; mean(e_list,1); std(e_list,1)]
                end
                for k = 1:order
                    [vtemp,dtemp] = eig(nmat(1:k,1:k)\hmat(1:k,1:k));
                    [notused,sorted] = sort(diag(dtemp));
                    vsmall(1:k,k) = vtemp(1:k,sorted(1));
                    overlap_list(accept_hmat,k) = ...
                        abs(vsmall_exact(1:k,k)'*nmat_exact(1:k,1:k)*vsmall(1:k,k)) ...
                        /sqrt(abs(vsmall_exact(1:k,k)'*nmat_exact(1:k,1:k)* ...
                        vsmall_exact(1:k,k) ...
                    *abs(vsmall(1:k,k)'*nmat_exact(1:k,1:k)*vsmall(1:k,k))));
                    max_order_overlap_list(accept_hmat,k) = ...
                        abs(vsmall_exact(1:order,order)'*nmat_exact(1:order,1:k)*vsmall(1:k,k)) ...
                        /sqrt(abs(vsmall_exact(1:order,order)'*nmat_exact(1:order,1:order)* ...
                        vsmall_exact(1:order,order) ...
                    *abs(vsmall(1:k,k)'*nmat_exact(1:k,1:k)*vsmall(1:k,k))));                
                end
                if (size(e_list,1) > 1)
                    [ones(1,order); overlap_start; mean(overlap_list,1); std(overlap_list,1)]
                    [max_order_overlap_exact; max_order_overlap_start; mean(max_order_overlap_list,1); ...
                        std(overlap_list,1)]                    
                end
            end
        end
    end
end
    

