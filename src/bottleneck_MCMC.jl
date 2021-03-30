# bottleneck.jl
#
# Bayesian inference of the number of SARS-CoV-2 virions
# that initiate COVID-19 based on sequence data of
# donor and recipient pairs using a virus bottleneck model
# Modification of model in Leonard et al. (2017).


"""
metropolis_hastings(total,r,d,alpha,lambda)

Metropolis-Hastings MCMC of model with hierarchical prior on bottleneck number
d = sequence variant counts of donor (one Vector per donor)
r = sequence variant counts of recipient (one Vector per recipient)
lambda = Hierarchichal Prior Poisson rate parameter for bottleneck virion number distribution
N = bottleneck number in a given recipient (Int)
b = bottleneck count vector of all variants (Vector)
epsilon = mutation/noise rate per variant (Vector)
alpha = prior exponential scale for epsilon
total = number of MCMC draws
temp = optional temperature to modify acceptance rate

    N ~ Poisson(lambda)
    b ~ DirichletMultinomial(N,d), sum(b) = N
    r ~ DirichletMultinomial(R_T,b + epsilon)
    epsilon[i] ~ Exponential(alpha)

"""

function metropolis_hastings(total,r,d,alpha,lambda::Int,temp=1.)
    b,epsilon = parameter_sample(d,alpha,lambda)
    metropolis_hastings(total,r,d,alpha,lambda,b,epsilon,temp)
end


function metropolis_hastings(total,r,d,alpha,lambda::Int,b::Vector,epsilon::Vector,temp=1.)
    Nout = Vector{Int}(undef,total)
    lambdaout = Vector{Int}(undef,total)
    bout = Matrix{Int}(undef,length(d),total)
    epsilonout = Matrix{Float64}(undef,length(d),total)
    N = sum(b)
    println(N)
    ll = loglikelihood(r,d,alpha,lambda,b,epsilon)
    J = logtransition(d,alpha,N,b,epsilon)
    println(ll)
    for t in 1:total
        lambdat = max(N_sample(lambda),1)
        Nt,bt,epsilont = parameter_samples(d,mean(epsilon),lambdat)
        llt = loglikelihood(r,d,alpha,lambdat,bt,epsilont)
        Jt = logtransition(d,mean(epsilon),lambdat,bt,epsilont)
        J = logtransition(d,mean(epsilont),lambda,b,epsilon)
        println(ll," ",llt," ",J," ",Jt," ",N," ",Nt)
        println(llt - ll - Jt + J)
        if log(rand()) < (llt - ll - Jt + J)/temp
            ll,N,b,epsilon,lambda = llt,Nt,bt,epsilont,lambdat
            println(N)
        end
        Nout[t] = N
        lambdaout[t] = lambda
        bout[:,t] = b
        epsilonout[:,t] = epsilon
    end
    println(ll)
    Nout,bout,epsilonout,lambdaout
end

"""
metropolis_hastings(total,r,d,alpha,b,epsilon,temp)

Metropolis-Hastings MCMC of a Bayesian model with uninformative prior on N
"""

function metropolis_hastings(total,r,d,alpha,b::Vector,epsilon::Vector,temp=1.)
    Nout = Vector{Int}(undef,total)
    bout = Matrix{Int}(undef,length(d),total)
    epsilonout = Matrix{Float64}(undef,length(d),total)
    N = sum(b)
    println(N)
    ll = loglikelihood(r,d,alpha,b,epsilon)
    J = logtransition(d,alpha,b,epsilon)
    println(ll)
    for t in 1:total
        Nt,bt,epsilont = parameter_samples(d,mean(epsilon),N)
        llt = loglikelihood(r,d,alpha,bt,epsilont)
        Jt = logtransition(d,mean(epsilon),bt,epsilont)
        J = logtransition(d,mean(epsilont),b,epsilon)
        # println(ll," ",llt," ",J," ",Jt," ",N," ",Nt)
        # println(llt - ll - Jt + J)
        if log(rand()) < (llt - ll - Jt + J)/temp
            ll,N,b,epsilon = llt,Nt,bt,epsilont
            # println(N)
        end
        Nout[t] = N
        bout[:,t] = b
        epsilonout[:,t] = epsilon
    end
    println(ll)
    Nout,bout,epsilonout
end


"""
mock_recipient_reads(R_t,d,alpha,N)

Generative model for recipient variant counts
given total variant counts R, N, d, and alpha
"""
function mock_recipient_reads(R,d,alpha,N)
    b,epsilon= parameter_sample(d,alpha,N)
    println(b)
    rand(dist_r(R,b,epsilon))
end
"""
parameter_samples(d,lambda,alpha)

Draw parameter samples for N, b, an epsilon
for MCMC proposal
"""
function parameter_samples(d,alpha,lambda)
    N = max(N_sample(lambda),1)
    b,epsilon = parameter_sample(d,alpha,N)
    N,b,epsilon
end

"""
N_sample(lambda)

Parameter samples for N with mean lambda
"""
N_sample(lambda) = rand(dist_N(lambda))

"""
parameter_sample(d,alpha,N)

samples for b and epsilon
"""
function parameter_sample(d,alpha,N)
    b = rand(dist_b(N,d))
    epsilon = rand(dist_epsilon(length(b),alpha))
    b,epsilon
end

"""
logtransition(d,alpha,N0,b,epsilon)
logtransition(d,alpha,b,epsilon)

Log of the proposal probability from d, alpha, No to b and epsilon

"""
logtransition(d,alpha,N0,b,epsilon) =  ll_b(b,d) + ll_epsilon(epsilon,alpha) + ll_N(sum(b),N0)
logtransition(d,alpha,b,epsilon) =  ll_b(b,d) + ll_epsilon(epsilon,alpha)

# Conditional Distributions for parameters

dist_N(lambda) = Poisson(lambda)
dist_b(N,d) = DirichletMultinomial(N,d)
function dist_epsilon(n,alpha::Vector)
    dv = Vector{Exponential}(undef,n)
    for i in eachindex(d)
        dv[i] = Exponential(alpha)
    end
    product_distribution(dv)
end
dist_r(R,b::Vector,epsilon::Vector) = DirichletMultinomial(R,b + epsilon)

"""
loglikelihood(r,d,alpha,lambda,b,epsilon)
loglikelihood(r,d,alpha,b,epsilon)

Log Likelihoods of models
"""
loglikelihood(r,d,alpha,lambda,b,epsilon) = ll_r(r,b,epsilon) + ll_b(b,d) + ll_epsilon(epsilon,alpha) + ll_N(sum(b),lambda)
loglikelihood(r,d,alpha,b,epsilon) = ll_r(r,b,epsilon) + ll_b(b,d) + ll_epsilon(epsilon,alpha)

# Log Likelihood functions
ll_N(N,lambda) = logpdf(dist_N(lambda),N)
ll_epsilon(epsilon,alpha) = logpdf(dist_epsilon(length(epsilon),alpha),epsilon)
ll_b(b,d) = logpdf(dist_b(sum(b),d),b)
ll_r(r,b,epsilon) = logpdf(dist_r(sum(r),b,epsilon),r)
