
% This is a matlab implementation of our AAAI 2018 paper:
% Zhiyong Yang, Qianqian Xu, Xiaochun Cao, Qingming Huang:From Common to Special: When Multi-Attribute Learning Meets Personalized Opinions. AAAI 2018: 515-522
%
% Acknowledement: We would like to thank Dr. Jiayu Zhou for providing the MALSAR package  https://github.com/jiayuzhou/MALSAR 


% If you use this code, please cite our paper with the following  bibtex code:
% @inproceedings{Yang2018FromCT,
%  title={From Common to Special: When Multi-Attribute Learning Meets Personalized Opinions},
%  author={Zhiyong Yang and Qianqian Xu and Xiaochun Cao and Qingming Huang},
%  booktitle={AAAI},
%  year={2018}
% }


%%%% Objective Function:
%  (1/2) * \sum || y^{(i,j)} - X^{(i,j)}(\theta + p^{(i)} + u^{(i,j)})||^2 
%      + \lambda_1 ||P||_{1,2} + \lambda_2 ||U^\top||_{1,2} + \lambda_3 ||\theta||_2^2 + 


%%%% @Parameters:
% X(Input feature): A cell with scale num_t * num_u, where num_t is the number of attributes, and num_u represents the number of annotaters for each attribute.
%                   Each element of X (X{i,j}) is an input feature matrix with scale num_sample * num_feature. 
% Y(Input label):  A cell with scale num_t * num_u, where num_t is the number of attributes, and num_u represents the number of annotaters for each attribute.
%                  Each element of Y (Y{i,j}) is an input label vector with scale num_sample * 1. 
% lambda1:        see the objective function
% lambda2:        see the objective function 
% lambda3:        see the objective function
%opts:            The same as MALSAR

function [model,funcVal] = MUTUAL_simp(X, Y, lambda1, lambda2,lambda3, opts)
if nargin <5
    error('\n Inputs: X, Y, and lambda1,  lambda2, lambda2 should be specified!\n');
end
if nargin <6
    opts = [];
end

% initialize options.
opts=init_opts(opts);

% initial Lipschiz constant. 
if isfield(opts, 'lFlag')
    lFlag = opts.lFlag;
else
    lFlag = false;
end

dimension = size(X{1,1}, 2);
[num_t, num_u] = size(Y) ;
nL = num_u * num_t;
% initialize a starting point
if opts.init==2
    P0 = zeros(dimension, num_t);
    U0 = zeros(dimension, num_u, num_t) ;
    theta0 = zeros(dimension, 1) ;
elseif opts.init == 0
    P0 = randn(dimension, num_t);
    U0 = randn(dimension, num_u, num_t) ;
    theta0 = randn(dimension, 1) ;
else
    if isfield(opts,'P0')
        P0=opts.P0;
        if (nnz(size(P0)-[dimension, num_t]))
            error('\n Check the input .P0');
        end
    else
        P0=zeros(dimension, num_t);
    end
    

    
    if isfield(opts,'U0')
        U0=opts.U0;
        if (nnz(size(U0)-[dimension, num_u, num_t]))
            error('\n Check the input .U0');
        end
    else
        U0=zeros(dimension, num_u, num_t);
    end
      
    if isfield(opts,'theta0')
        theta0=opts.theta0;
        if (nnz(size(theta0)-[dimension]))
            error('\n Check1 the input .theta0');
        end
    else
        theta0=zeros(dimension, num_u, num_t);
    end
end
d = dimension ;
coef = 1 ;
% Set an array to save the objective value
funcVal = [];
%% initialization and precalculation
P = P0;
U = U0;
theta = theta0;

XtX = cellfun(@(elem) elem' * elem, X, 'UniformOutput', false ); 
XtY = cellfun(@(e1, e2) e1' * e2, X, Y, 'UniformOutput', false);


Pn = P;  Un  = U; thetan = theta ;
t_new = 1; 
% bound L 
% L1norm = max( sum(abs(X),1) ); Linfnorm = max( sum(abs(X),2) );
C = sqrt( nL* 6/ coef) ;
if lFlag
    % Upper bound for largest eigenvalue of Hessian matrix
%     L = C * min( [L1norm * Linfnorm;...
%                   size(X,1) * Linfnorm * Linfnorm; ...
%                   size(X,2) * L1norm * L1norm; ...
%                   size(X,1) * size(X,2) * max( abs(X(:)) ) ] ...
%                );
else
    % Lower bound for largest eigenvalue of Hessian matrix
%     L = C * min( L1norm * L1norm / size(X,1), Linfnorm * Linfnorm / size(X,2) );
    L = 1 ;
end
% Initial function value
funcVal = cat(1, funcVal, eval_loss());

%count = 0;
for iter = 1:opts.maxIter
    P_old = P; U_old = U ; theta_old = theta ;
    t_old = t_new;
  
    grad_P = zeros(d, num_t);
    grad_U = zeros(d, num_u, num_t);
    grad_theta = zeros(d, 1) ;
    for nt  = 1:num_t
        for nu = 1:num_u
          delta =  Pn(:, nt)  + theta + Un(:, nu, nt);
          delta =  2 * coef * (XtX{nt, nu} * delta  - XtY{nt, nu}) ;
          grad_P(:, nt)     = grad_P(:, nt) + delta;
          grad_theta        = grad_theta  + delta ;
          grad_U(:, nu, nt) =  delta ;
        end
    end 
    % If we estimate the upper bound of Lipschitz constant, no line search
    % is needed.
    if lFlag
        update_param() ;
    else
        % line search 
        for inneriter = 1:20
            update_param() ;
            dP = P - Pn;   dU = U - Un ; dtheta = theta - thetan ;
            Lhs = 0;
            
            for i = 1 :num_t
                for j = 1:num_u
                    dW = dP(:, i) + dtheta +  dU(:, j, i) ;
                    Lhs = Lhs + 2 * coef * dW' * XtX{i,j} * dW ;
                end
            end
            
            Rhs = L * ( sumsqr(dP) + sumsqr(dU) + sumsqr(dtheta) ) ; 
            if  Lhs <= Rhs
                break;
            else
                L = L*1.4;
            end
        end
    end
    funcVal = cat(1, funcVal, eval_loss()); 
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end

    % Update the coefficient
    t_new = ( 1 + sqrt( 1 + 4 * t_old^2 ) ) / 2;
    dt    = (t_old-1) / t_new ;
    % Update reference points 
    Pn    = P + dt * (P - P_old);
  
    Un    = U + dt * (U - U_old);
    
    thetan = theta + dt * (theta - theta_old) ;
end

W = cell(num_t, num_u) ;
for n1 = 1: num_t
    for n2  = 1 :num_u
        W{n1, n2} = P(:, n1)  + theta + U(:, n2, n1) ;
    end
end

model.W = W ;
model.U = U ;
model.P = P ;
model.theta =theta ;

    function update_param()
        P = proximalL12norm(Pn - grad_P/L, lambda1/L);
        for t = 1:num_t
            
            U(:, :, t) = proximalL12norm(Un(:, :, t)' - grad_U(:, :, t)'/L, lambda2/L)';
        end
        theta = proximalL1norm(thetan - grad_theta/L, lambda3/L) ;
    end 

    function l = eval_loss()
        l1 =  lambda1 * L12norm(P) + lambda3 * l1norm(theta)  ;
        lu = 0;
        for tt = 1:num_t
            lu = lu + lambda2 * L12norm(U(:, :, tt)') ;
        end
        
        l1 = l1 + lu ;
        l2 = 0;
        for i1 = 1: num_t
            for i2 = 1:num_u
             para  = P(: , i1)+ U(:, i2, i1) + theta;
             l2 = l2 + coef * sumsqr(Y{i1, i2} - X{i1, i2} * para) ;
            end
        end
        
        l = l1 + l2 ;
    end
end

