function [ res ] = proximalL1norm( X, thresh )
%PROXIMALL1NORM Summary of this function goes here
%   Detailed explanation goes here
res = max(abs(X) -thresh, 0) .* sign(X) ;

end

