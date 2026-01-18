#=
================================================================================
Hamiltonian Monte Carlo (HMC) Sampler for Latent Traits
================================================================================
Samples missing/latent omics values in the NNMM framework using HMC.

Network Architecture:
                   |---- Z[:,1] ----- Z0*W0[:,1]
    yobs ---f(X)---|---- Z[:,2] ----- Z0*W0[:,2]
                   |---- Z[:,3] ----- Z0*W0[:,3]

    Where f(X) is the activation function (tanh, sigmoid, etc.)

Notation:
  X  : Marker covariate matrix, n × p (each column = 1 marker)
  Z  : Latent traits, n × l1 (each column = 1 latent trait)
  y  : Observed trait, vector of length n
  W0 : Marker effects, p × l1 (each column = effects for 1 latent trait)
  W1 : Weights from hidden layer to output, vector of length l1
  Mu0: Bias terms for latent traits, vector of length l1
  mu : Bias for observed trait, scalar
  Sigma2z: Residual variance of latent traits, diagonal l1 × l1
  sigma2e: Residual variance of observed trait, scalar

Key Functions:
  calc_gradient_z: Compute gradient for HMC proposal
  calc_log_p_z: Compute log probability for acceptance ratio
  hmc_one_iteration_z!: Perform one HMC step

Reference:
  Neal (2011) MCMC using Hamiltonian dynamics. 
  Handbook of Markov Chain Monte Carlo.

Author: NNMM.jl Team
================================================================================
=#


# Fast elementwise activation derivative for built-in activations (avoids ForwardDiff).
function activation_derivative(activation_function, x)
    fname = string(nameof(typeof(activation_function)))
    T = eltype(x)

    if occursin("mylinear", fname)
        return fill(one(T), size(x))
    elseif occursin("mytanh", fname)
        y = activation_function.(x)
        return one(T) .- y .^ 2
    elseif occursin("mysigmoid", fname)
        y = activation_function.(x)
        return y .* (one(T) .- y)
    elseif occursin("myrelu", fname)
        return ifelse.(x .> zero(T), one(T), zero(T))
    elseif occursin("myleakyrelu", fname)
        return ifelse.(x .> zero(T), one(T), T(0.01))
    else
        return ForwardDiff.derivative.(activation_function, x)
    end
end

#helper 1: calculate gradiant of all latent traits for all individual
function calc_gradient_z(ylats,yobs,weights_NN,σ_ylats,σ_yobs,ycorr,activation_function,ycorr_yobs) #ycorr is 1->2, ycorr_yobs is 2->3
    # μ1, w1     = weights_NN[1], weights_NN[2:end]
    w1 = weights_NN
    # g_ylats = activation_function.(ylats)
    g_ylats_derivative = activation_derivative(activation_function, ylats)
    dlogf_ylats    = -ycorr / σ_ylats
    # dlogfy         = ((yobs .- μ1 - g_ylats*w1)/σ_yobs) * w1' .* g_ylats_derivative #size: (n, l1)
    dlogfy         = (ycorr_yobs/σ_yobs) * w1' .* g_ylats_derivative #size: (n, l1)
    gradient_ylats = dlogf_ylats + dlogfy

    return gradient_ylats  #size (n,l1)
end

# helper 2: calculate log p(z|y) to help calculate the acceptance rate
function calc_log_p_z(ylats,yobs,weights_NN,σ_ylats,σ_yobs,ycorr,activation_function,ycorr_yobs) #ycorr is 1->2, ycorr_yobs is 2->3
    # μ1  = weights_NN[1]
    # w1 = weights_NN[2:end]
    # g_ylats = activation_function.(ylats)
    quad_form = sum((ycorr / σ_ylats) .* ycorr, dims=2)
    logdet_σ_ylats = σ_ylats isa Number ? log(σ_ylats) : logdet(Symmetric(σ_ylats))
    logf_ylats = -0.5 .* quad_form .- (0.5 * logdet_σ_ylats)
    # logfy      = -0.5*(yobs .- μ1 - g_ylats*w1).^2 /σ_yobs .- 0.5*log(σ_yobs)
    logfy      = -0.5*(ycorr_yobs).^2 /σ_yobs .- 0.5*log(σ_yobs)
    log_p_ylats= logf_ylats + logfy

    return log_p_ylats  #size: (n,1)
end

#helper 3: one iterations of HMC to sample Z
#ycor is a temporary variable to save ycorr after reshape; ycorr is residual for latent traits
function hmc_one_iteration(nLeapfrog,ϵ,ylats_old,yobs,weights_NN,σ_ylats,σ_yobs,ycorr,activation_function,ycorr_yobs)
    nobs, ntraits  = size(ylats_old)
    ylats_old = copy(ylats_old)
    ylats_new = copy(ylats_old)
    is_linear_activation = occursin("mylinear", string(nameof(typeof(activation_function))))
    T = eltype(ylats_new)
    ϵT = T(ϵ)

    #step 1: Initiate Φ from N(0,M)
    Φ = randn(T, nobs, ntraits) #rand(n,Normal(0,M=1.0)), tuning parameter: M
    log_p_old = calc_log_p_z(ylats_old,yobs,weights_NN,σ_ylats,σ_yobs,ycorr,activation_function,ycorr_yobs) - 0.5*sum(Φ.^2,dims=2)  #(n,1)
    #step 2: update (ylats,Φ) from 10 leapfrog
    #2(a): update Φ
    Φ += (ϵT/2) * calc_gradient_z(ylats_new,yobs,weights_NN,σ_ylats,σ_yobs,ycorr,activation_function,ycorr_yobs)  #(n,l1)
    for leap_i in 1:nLeapfrog
       #2(b) update latent traits
       ylats_tmp = copy(ylats_new) #ylat before update
       ylats_new += ϵT * Φ  # (n,l1)
       ycorr     += ϵT * Φ  #update ycorr due to change of Z
       if is_linear_activation
           ycorr_yobs += (ylats_tmp - ylats_new) * weights_NN
       else
           ycorr_yobs += (activation_function.(ylats_tmp)-activation_function.(ylats_new))*weights_NN #update ycorr_yobs due to change of Z
       end
       #(c) half step of phi
       if leap_i == nLeapfrog
           #2(c): update Φ
           Φ += (ϵT/2) * calc_gradient_z(ylats_new,yobs,weights_NN,σ_ylats,σ_yobs,ycorr,activation_function,ycorr_yobs)
       else
           #2(a)+2(c): update Φ
           Φ += ϵT * calc_gradient_z(ylats_new,yobs,weights_NN,σ_ylats,σ_yobs,ycorr,activation_function,ycorr_yobs)
       end
    end

    #Step3. acceptance rate
    log_p_new = calc_log_p_z(ylats_new,yobs,weights_NN,σ_ylats,σ_yobs,ycorr,activation_function,ycorr_yobs) - 0.5*sum(Φ.^2,dims=2) #(n,1)
    r         = exp.(log_p_new - log_p_old)  # (n,1)
    nojump    = rand(T, nobs) .> r  # bool (n,1)

    for i in 1:nobs
        if nojump[i]
            ylats_new[i,:] = ylats_old[i,:]
        end
    end

    return ylats_new
end

"""
    sample_latent_traits_linear_gaussian(μ_ylats, y_centered, weights_NN, σ_ylats, σ_yobs)

Sample latent traits (omics) for a *linear* activation NNMM update in closed form.

This avoids HMC instability in the linear/Gaussian case where the conditional distribution
`p(z | y, ...)` is multivariate Normal.

Inputs:
- `μ_ylats`: `n×l` matrix of prior means for each individual (from the 1→2 model).
- `y_centered`: length-`n` vector of `y - Xb` (2→3 non-marker effects removed).
- `weights_NN`: length-`l` vector of omics→phenotype weights.
- `σ_ylats`: latent-trait residual covariance (Number / `Diagonal` / `AbstractMatrix`).
- `σ_yobs`: phenotype residual variance (scalar).

Returns:
- `n×l` matrix of sampled latent traits.
"""
function sample_latent_traits_linear_gaussian(μ_ylats, y_centered, weights_NN, σ_ylats, σ_yobs)
    nobs, ntraits = size(μ_ylats)
    if nobs == 0
        return copy(μ_ylats)
    end

    T = eltype(μ_ylats)
    w = T.(weights_NN)
    y = T.(y_centered)

    Σw = if σ_ylats isa Number
        T(σ_ylats) .* w
    else
        σ_ylats * w
    end

    denom = float(σ_yobs) + float(dot(w, Σw))
    if !isfinite(denom) || denom <= 0
        error("NNMM: invalid denom for linear latent update: denom=$denom σ_yobs=$σ_yobs")
    end

    Σ = if σ_ylats isa Number
        Matrix{T}(I, ntraits, ntraits) .* T(σ_ylats)
    else
        Matrix{T}(σ_ylats)
    end

    cov = Σ .- (Σw * Σw') ./ T(denom)
    cov = Symmetric(cov)

    # Cholesky can fail if `cov` is numerically near-singular; add a small jitter.
    L = nothing
    jitter = zero(T)
    for attempt in 1:6
        try
            L = cholesky(cov + jitter * I).L
            break
        catch
            jitter = attempt == 1 ? eps(T) : jitter * T(10)
        end
    end
    if L === nothing
        error("NNMM: failed to factor linear latent covariance matrix (cov not PD)")
    end

    μ_dot = μ_ylats * w
    resid = y .- μ_dot
    μ_post = μ_ylats .+ (resid ./ T(denom)) * Σw'

    Z = randn(T, nobs, ntraits)
    ylats_new = μ_post .+ Z * L'
    if any(x -> !isfinite(x), ylats_new)
        error("NNMM: non-finite draw in linear latent update")
    end
    return ylats_new
end
