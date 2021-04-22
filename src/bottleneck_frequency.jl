# bottleneck.jl
#
# Bayesian inference of the number of SARS-CoV-2 virions
# that initiate COVID-19 based on sequence data of
# donor and recipient pairs using a virus bottleneck model
# Find stationary variant frequency distribution using
# powerlaw mutation distribution and
# transmission model of Leonard et al. (2017).

"""
logposterior(Nmax,data_pdf,bins)

Compute logposterior for bottleneck number N
using noninformative prior

N = [0,Nmax]
data_pdf = frequency density of data
bins = frequency bins
"""
function logposterior1(Nmax,data_pdf,bins,w)
    p = Vector{Float64}(undef,Nmax)
    for N in 1:Nmax
        p[N] = neg_crossentropy(data_pdf,ll(bins,log.(ll(bins,N,w)),N,w))
    end
    p .- logsumexp(p)
end

function logposterior1(Nmax,data_pdf,bins)
    p = zeros(Nmax,10)
    for N in 1:Nmax, w in 1:10
        p[N,w] = neg_crossentropy(data_pdf,ll(bins,log.(ll(bins,N,w*.1)),N,w*.1))
    end
    pN = sum(p,dims=2)[:,1]
    pW = sum(p,dims=1)[1,:]
    return pN .- logsumexp(pN), pW .- logsumexp(pW)
end


function logposterior(Nmax,rpmf,dpmf,binsd)
    p = zeros(Nmax,10)
    for N in 1:Nmax, w in 1:10
        p[N,w] = neg_crossentropy(rpmf,ll(binsd,log.(dpmf),N,w*.1))
    end
    pN = sum(p,dims=2)[:,1]
    pW = sum(p,dims=1)[1,:]
    return pN .- logsumexp(pN),pW .- logsumexp(pW)
end

function logposterior(Nmax,rpmf,dpmf,binsd,w)
    p = zeros(Nmax)
    for N in 1:Nmax
        p[N] = neg_crossentropy(rpmf,ll(binsd,log.(dpmf),N,w))
    end
    return pN .- logsumexp(pN)
end

"""
neg_crossentropy(f,g)


"""
neg_crossentropy(f,g) = f' * log.(max.(g,eps(Float16)))


"""


"""
ll(f,N,w) = ll(f,logpdf_powerlaw.(f,2.4),N,w)

function ll(f,lp,N,w)
    z=logpdf_transmission(f,lp,N)
    y = w*exp.(lp) + (1-w)*exp.(z)
    y /=sum(y)
    # noise_poisson(exp.(lp)+exp.(z))
end


"""


"""
function logpdf_transmission(f,lp,N)
    pt = similar(lp)
    a = similar(lp)
    for i in eachindex(f)
        for j in eachindex(f)
            a[j] = transmission_matrix(f[i],f[j],N) + lp[j]
        end
        pt[i] = check_nan(logsumexp(a))
    end
    return pt
end

"""

"""
function transmission_matrix(fr,fd,N)
    a = Vector{Float64}(undef,N+1)
    for k in 0:N
        a[k+1] = logpmf_r(round(Int,fr*100),100,k,N) + logpmf_k(k,N,fd)
        # a[k+1] = logpmf_r(fr,k,N) + logpmf_k(k,N,fd)
    end
    check_nan(logsumexp(a))
end

logpmf_k(k,N,p) = logpdf(Binomial(N,p),k)
logpmf_r(f,k,N) = logpdf(Beta(max(k,eps(Float64)),max(N-k,eps(Float64))),f)
logpmf_r(r,R,k,N) = logpdf(dist_r(R,k,N),r)
dist_r(R,k,N) = BetaBinomial(R,max(k,eps(Float64)),max(N-k,eps(Float64)))

"""
check_nan(a)

"""
function check_nan(a)
    isnan(a) ? -Inf : a
end

"""
logpdf_powerlaw(f,alpha) = -alpha * log(f)


"""
logpdf_powerlaw(f,alpha) = -alpha * log(f)


"""
logsumexp(a)


"""
function logsumexp(a)
    c = maximum(a)
    if c == Inf
        return Inf
    else
        return c + log(sum(exp.(a .- c)))
    end
end


function noise_poisson(p)
    p1 = zeros(length(p))
    for m in eachindex(p)
        d = Poisson(m-1)
        for c in eachindex(p1)
            p1[c] += p[m]*pdf(d,c-1)
        end
    end
    p1
end

function noise(p,lambda)
    d = Poisson(lambda)
    a = zeros(length(p))
    for i in eachindex(p)
        for j in 0:i
            a[i] += pdf(d,j)*p[i-j]
        end
    end
    return a
end
