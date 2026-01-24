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

# Masked HMC update: only update missing entries, conditioning on observed middle-layer values.
#
# `observed_mask[i,j] == true` means the value `ylats_old[i,j]` is observed/fixed and should NOT be updated.
# The HMC dynamics are run in the subspace of missing coordinates, but the log-density is computed on the
# full vector so the update targets p(z_miss | z_obs, y, ...).
function hmc_one_iteration_masked(
    nLeapfrog,
    ϵ,
    ylats_old,
    yobs,
    weights_NN,
    σ_ylats,
    σ_yobs,
    ycorr,
    activation_function,
    ycorr_yobs,
    observed_mask,
)
    nobs, ntraits = size(ylats_old)
    if size(observed_mask) != (nobs, ntraits)
        error("NNMM: observed_mask size mismatch in masked HMC latent update")
    end

    ylats_old = copy(ylats_old)
    ylats_new = copy(ylats_old)
    is_linear_activation = occursin("mylinear", string(nameof(typeof(activation_function))))
    T = eltype(ylats_new)
    ϵT = T(ϵ)

    missing_mask = .!observed_mask

    # step 1: sample momentum only for missing coordinates
    Φ = randn(T, nobs, ntraits)
    Φ .*= missing_mask
    log_p_old = calc_log_p_z(
        ylats_old,
        yobs,
        weights_NN,
        σ_ylats,
        σ_yobs,
        ycorr,
        activation_function,
        ycorr_yobs,
    ) .- 0.5 .* sum(Φ .^ 2, dims=2)

    # step 2: leapfrog updates (restricted to missing coordinates)
    Φ .+= (ϵT / 2) .* calc_gradient_z(
        ylats_new,
        yobs,
        weights_NN,
        σ_ylats,
        σ_yobs,
        ycorr,
        activation_function,
        ycorr_yobs,
    ) .* missing_mask

    for leap_i in 1:nLeapfrog
        ylats_tmp = copy(ylats_new)
        ylats_new .+= ϵT .* Φ
        ycorr .+= ϵT .* Φ

        if is_linear_activation
            ycorr_yobs .+= (ylats_tmp - ylats_new) * weights_NN
        else
            ycorr_yobs .+= (activation_function.(ylats_tmp) .- activation_function.(ylats_new)) * weights_NN
        end

        grad = calc_gradient_z(
            ylats_new,
            yobs,
            weights_NN,
            σ_ylats,
            σ_yobs,
            ycorr,
            activation_function,
            ycorr_yobs,
        )
        if leap_i == nLeapfrog
            Φ .+= (ϵT / 2) .* grad .* missing_mask
        else
            Φ .+= ϵT .* grad .* missing_mask
        end
    end

    # step 3: Metropolis accept/reject
    log_p_new = calc_log_p_z(
        ylats_new,
        yobs,
        weights_NN,
        σ_ylats,
        σ_yobs,
        ycorr,
        activation_function,
        ycorr_yobs,
    ) .- 0.5 .* sum(Φ .^ 2, dims=2)

    r = exp.(log_p_new .- log_p_old) # (n,1)
    nojump = rand(T, nobs) .> r

    for i in 1:nobs
        if nojump[i]
            ylats_new[i, :] = ylats_old[i, :]
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

"""
    sample_missing_latent_traits_linear_gaussian!(ylats, μ_ylats, y_centered, weights_NN, σ_ylats, σ_yobs, observed_mask)

Sample *missing* latent-trait/omics entries for a linear-activation NNMM update.

Model for each individual `i`:
- Prior (from 1→2 layer): `z_i ~ Normal(μ_i, σ_ylats)`
- Likelihood (2→3, linear): `y_centered_i ~ Normal(z_i' * weights_NN, σ_yobs)`

Entries with `observed_mask[i,j] == true` are treated as fixed/observed and are **not** updated.
Missing entries are sampled from the exact Gaussian conditional
`p(z_miss | z_obs, y_centered, ...)`.

`y_centered` must be `y - Xb` (i.e., phenotype with all non-omics effects removed) and aligned
to the rows of `ylats`/`μ_ylats`.

Updates `ylats` in-place and returns it.
"""
function sample_missing_latent_traits_linear_gaussian!(
    ylats::AbstractMatrix,
    μ_ylats::AbstractMatrix,
    y_centered::AbstractVector,
    weights_NN::AbstractVector,
    σ_ylats,
    σ_yobs,
    observed_mask::AbstractMatrix
)
    nobs, ntraits = size(μ_ylats)
    if nobs == 0
        return ylats
    end

    if size(ylats) != (nobs, ntraits)
        error("NNMM: ylats size mismatch in linear latent update")
    end
    if length(y_centered) != nobs
        error("NNMM: y_centered length mismatch in linear latent update")
    end
    if length(weights_NN) != ntraits
        error("NNMM: weights_NN length mismatch in linear latent update")
    end
    if size(observed_mask) != (nobs, ntraits)
        error("NNMM: observed_mask size mismatch in linear latent update")
    end

    T = eltype(μ_ylats)
    w = T.(weights_NN)
    y = T.(y_centered)
    σy = T(σ_yobs)
    if !isfinite(σy) || σy <= 0
        error("NNMM: invalid σ_yobs in linear latent update: σ_yobs=$σy")
    end

    # Fast path: rows where all traits are unobserved (i.e., fully latent) can be sampled in closed form.
    n_observed_per_row = vec(sum(observed_mask, dims=2))
    fully_latent = n_observed_per_row .== 0
    if any(fully_latent)
        fully_latent_rows = findall(fully_latent)
        ylats[fully_latent_rows, :] = sample_latent_traits_linear_gaussian(
            μ_ylats[fully_latent_rows, :],
            y[fully_latent_rows],
            w,
            σ_ylats,
            σy,
        )
    end

    # Remaining rows: partially observed traits. Sample only missing entries conditional on observed ones.
    partial_rows = findall(n_observed_per_row .> 0)
    if isempty(partial_rows)
        return ylats
    end

    Σ = if σ_ylats isa Number
        Matrix{T}(I, ntraits, ntraits) .* T(σ_ylats)
    else
        Matrix{T}(σ_ylats)
    end

    if ntraits <= 64
        groups = Dict{UInt64, Vector{Int}}()
        for i in partial_rows
            if all(view(observed_mask, i, :))
                continue
            end
            mask = zero(UInt64)
            @inbounds for j in 1:ntraits
                if observed_mask[i, j]
                    mask |= (UInt64(1) << (j - 1))
                end
            end
            push!(get!(groups, mask, Int[]), i)
        end

        for (mask, rows) in groups
            obs_idx = Int[]
            miss_idx = Int[]
            @inbounds for j in 1:ntraits
                if (mask >> (j - 1)) & UInt64(1) == UInt64(1)
                    push!(obs_idx, j)
                else
                    push!(miss_idx, j)
                end
            end
            if isempty(miss_idx)
                continue
            end

            w_m = w[miss_idx]
            w_o = w[obs_idx]

            Σ_mm = Σ[miss_idx, miss_idx]
            Σ_cond = Σ_mm
            Σ_mo = isempty(obs_idx) ? zeros(T, length(miss_idx), 0) : Σ[miss_idx, obs_idx]
            Σ_oo_factor = nothing

            if !isempty(obs_idx)
                Σ_oo = Σ[obs_idx, obs_idx]
                jitter = zero(T)
                for attempt in 1:6
                    try
                        Σ_oo_factor = cholesky(Symmetric(Σ_oo + jitter * I))
                        break
                    catch
                        jitter = attempt == 1 ? eps(T) : jitter * T(10)
                    end
                end
                if Σ_oo_factor === nothing
                    error("NNMM: failed to factor Σ_oo in linear conditional latent update")
                end

                Σ_oo_inv_Σ_om = Σ_oo_factor \ Σ_mo'
                Σ_cond = Σ_mm .- Σ_mo * Σ_oo_inv_Σ_om
            end

            Σ_cond = Symmetric(Σ_cond)
            Σw = Σ_cond * w_m
            denom = σy + dot(w_m, Σw)
            if !isfinite(denom) || denom <= 0
                error("NNMM: invalid denom for conditional linear latent update: denom=$denom σ_yobs=$σy")
            end

            cov = Matrix(Σ_cond) .- (Σw * Σw') ./ T(denom)
            cov = Symmetric(cov)

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
                error("NNMM: failed to factor conditional linear latent covariance matrix (cov not PD)")
            end

            z_obs = ylats[rows, obs_idx]
            μ_o = μ_ylats[rows, obs_idx]
            μ_m = μ_ylats[rows, miss_idx]

            μ_cond = μ_m
            if !isempty(obs_idx)
                delta = z_obs .- μ_o # n×o
                tmp = Σ_oo_factor \ delta' # o×n
                μ_cond = μ_m .+ (Σ_mo * tmp)' # n×k
            end

            y_eff = y[rows] .- z_obs * w_o
            resid = y_eff .- μ_cond * w_m
            μ_post = μ_cond .+ (resid ./ T(denom)) * Σw'

            Z = randn(T, length(rows), length(miss_idx))
            ylats_draw = μ_post .+ Z * L'
            if any(x -> !isfinite(x), ylats_draw)
                error("NNMM: non-finite draw in conditional linear latent update")
            end

            ylats[rows, miss_idx] = ylats_draw
        end
    else
        # Fallback: per-row conditional draws (avoids building huge grouping keys).
        for i in partial_rows
            obs_idx = findall(view(observed_mask, i, :))
            miss_idx = findall(x -> !x, view(observed_mask, i, :))
            if isempty(miss_idx)
                continue
            end

            z_obs = view(ylats, i, obs_idx)
            μ_o = view(μ_ylats, i, obs_idx)
            μ_m = view(μ_ylats, i, miss_idx)
            w_o = w[obs_idx]
            w_m = w[miss_idx]

            Σ_mm = Σ[miss_idx, miss_idx]
            Σ_mo = Σ[miss_idx, obs_idx]
            Σ_cond = Σ_mm
            μ_cond = copy(μ_m)
            if !isempty(obs_idx)
                Σ_oo = Σ[obs_idx, obs_idx]
                Σ_oo_factor = cholesky(Symmetric(Σ_oo))
                tmp = Σ_oo_factor \ (collect(z_obs) .- collect(μ_o))
                μ_cond .+= Σ_mo * tmp
                Σ_cond = Σ_mm .- Σ_mo * (Σ_oo_factor \ Σ_mo')
            end

            Σ_cond = Symmetric(Matrix(Σ_cond))
            Σw = Σ_cond * w_m
            denom = σy + dot(w_m, Σw)
            if !isfinite(denom) || denom <= 0
                error("NNMM: invalid denom for conditional linear latent update: denom=$denom σ_yobs=$σy")
            end

            cov = Matrix(Σ_cond) .- (Σw * Σw') ./ T(denom)
            cov = Symmetric(cov)
            L = cholesky(cov + eps(T) * I).L

            y_eff = y[i] - dot(w_o, z_obs)
            resid = y_eff - dot(w_m, μ_cond)
            μ_post = μ_cond .+ (resid / T(denom)) .* Σw

            draw = μ_post .+ (L * randn(T, length(miss_idx)))
            if any(x -> !isfinite(x), draw)
                error("NNMM: non-finite draw in conditional linear latent update")
            end
            ylats[i, miss_idx] = draw
        end
    end

    return ylats
end
