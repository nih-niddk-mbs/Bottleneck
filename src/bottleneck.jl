# bottleneck.jl
#
# Bayesian inference of the number of SARS-CoV-2 virions
# that initiate COVID-19 based on sequence data of
# donor and recipient pairs using a virus bottleneck model
# Modification of model in Leonard et al. (2017).

"""
likelihood_im(r,R,d,D,Nmax)

Compute likelihood of single variant of
a site for a single donor receptor pair
by convolving P(r|R,N,k) over
bottleneck distribution P(k|N,d,D)

r = variant counts at site i for recipient m
R = total counts at site i for recipient m
d = variant counts at site i for donor m
D = total at i for donor m
Nmax = maximum of N for sum
"""
function likelihood_im(r,R,d,D,Nmax)
    P = zeros(Nmax,2)
    for epsilon in 0:1
        for N in 1:Nmax
            for k in 0:N
                a = pmf_r(r,R,k+epsilon,N) * pmf_k(k,N,d,D)
                if isfinite(a)
                    P[N,epsilon+1] += a
                end
            end
        end
    end
    P
end

function loglikelihood_im(r,R,d,D,Nmax)
    P = likelihood_im(r,R,d,D,Nmax)
    for i in eachindex(P)
        if P[i] > 0
            P[i] = log(P[i])
        else
            P[i] = log(eps(BigFloat))
        end
    end
    P
end
"""
likelihood_m(r,R,d,D,Nmax)

Collate variant loglikelihoods of recipient-donor pair
r = Vector of variant counts at site i for recipient
R = Vector of total counts at site i for recipient
d = Vector of variant counts at site i for donor
D = Vector of total counts at site i for donor
"""
function loglikelihood_m(r,R,d,D,Nmax)
    llm = zeros(Nmax,2)
    for i in eachindex(r)
        llm += loglikelihood_im(r[i],R[i],d[i],D[i],Nmax)
    end
    llm
end

"""
loglikelihood(llm)

"""
function loglikelihood(llm)
    ll = zeros(size(llm[1]))
    for ll in llm
        ll += llm
    end
    ll
end

# pmf functions
pmf_epsilon(epsilon,alpha) = pdf(dist_epsilon(alpha),epsilon)
pmf_k(k,N,d,D) = pdf(dist_k(N,d,D),k)
pmf_r(r,R,k,N) = pdf(dist_r(R,k,N),r)

# Conditional Distributions for parameters
dist_k(N,d,D) = BetaBinomial(N,max(d,eps(BigFloat)),max(D-d,eps(BigFloat)))
dist_epsilon(alpha::Float64) = Bernoulli(alpha)
dist_r(R,k::Int,N::Int) = BetaBinomial(R,max(k,eps(BigFloat)),max(N-k,eps(BigFloat)))
