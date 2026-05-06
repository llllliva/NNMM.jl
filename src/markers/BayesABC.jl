#=
================================================================================
Single-Trait Bayesian Alphabet Samplers (BayesA, BayesB, BayesC)
================================================================================
Gibbs samplers for marker effect estimation in single-trait genomic prediction.

Methods:
- BayesA: All markers included with locus-specific variances
- BayesB: Variable selection with locus-specific variances
- BayesC: Variable selection with common variance

Key Functions:
- megaBayesABC!: Multi-threaded sampler for independent traits (constraint=true)
- BayesABC!: Core single-trait Gibbs sampler

Reference:
  Meuwissen et al. (2001) Prediction of total genetic value using 
  genome-wide dense marker maps. Genetics 157:1819-1829.

Author: NNMM.jl Team (adapted from JWAS.jl)
================================================================================
=#

"""
    megaBayesABC!(genotypes, wArray, vare, locus_effect_variances)

Multi-threaded BayesABC sampler for independent traits.

Used when `constraint=true` (diagonal covariance), allowing parallel
sampling across traits.
"""
function megaBayesABC!(genotypes, wArray, vare, locus_effect_variances; rngs=nothing)
    Threads.@threads for i in 1:length(wArray) #ntraits
        tid = Threads.threadid()
        rng = (rngs === nothing || length(rngs) < tid) ? Random.default_rng() : rngs[tid]
        BayesABC!(genotypes.mArray, genotypes.mRinvArray, genotypes.mpRinvm,
                  wArray[i], genotypes.α[i], genotypes.β[i], genotypes.δ[i], vare[i,i],
                  [vari[i,i] for vari in locus_effect_variances], genotypes.π[i]; rng=rng)
    end
end


function BayesABC!(genotypes, ycorr, vare, locus_effect_variances; rng=Random.default_rng())
    BayesABC!(genotypes.mArray,genotypes.mRinvArray,genotypes.mpRinvm,
              ycorr,genotypes.α[1],genotypes.β[1],genotypes.δ[1],vare,
              locus_effect_variances,genotypes.π; rng=rng)
end

function BayesABC!(xArray,xRinvArray,xpRinvx,
                   yCorr,
                   α,β,δ,
                   vare,varEffects,π; rng=Random.default_rng())

    logPi         = log(π)
    logPiComp     = log1p(-π)
    logDelta0     = logPi
    invVarRes     = 1/vare
    invVarEffects = 1 ./  varEffects
    logVarEffects = log.(varEffects)
    nMarkers      = length(α)

    for j=1:nMarkers
        x, xRinv = xArray[j], xRinvArray[j]
        rhs = (dot(xRinv,yCorr) + xpRinvx[j]*α[j])*invVarRes
        lhs = xpRinvx[j]*invVarRes + invVarEffects[j]
        invLhs = 1/lhs
        gHat   = rhs*invLhs
        logDelta1  = -0.5*(log(lhs) + logVarEffects[j] - gHat*rhs) + logPiComp
        probDelta1 = 1/(1+ exp(logDelta0 - logDelta1))
        oldAlpha = α[j]

        if rand(rng) < probDelta1
            δ[j] = 1
            β[j] = gHat + randn(rng)*sqrt(invLhs)
            α[j] = β[j]
            BLAS.axpy!(oldAlpha-α[j],x,yCorr)
        else
            if (oldAlpha!=0)
                BLAS.axpy!(oldAlpha,x,yCorr)
            end
            δ[j] = 0
            β[j] = randn(rng)*sqrt(varEffects[j])
            α[j] = 0
        end
    end
end


function BayesABC_block!(genotypes, ycorr, vare, locus_effect_variances; rng=Random.default_rng())
    BayesABC_block!(genotypes.MArray,genotypes.MRinvArray,genotypes.mpRinvm,
              genotypes.genotypes,genotypes.MpRinvM,
              ycorr,genotypes.α[1],genotypes.β[1],genotypes.δ[1],vare,
              locus_effect_variances,genotypes.π; rng=rng)
end

function BayesABC_block!(XArray,XRinvArray,xpRinvx,
                   X, XpRinvX,
                   yCorr,
                   α,β,δ,
                   vare,varEffects,π; rng=Random.default_rng())

    logPi         = log(π)
    logPiComp     = log1p(-π)
    logDelta0     = logPi
    invVarRes     = 1/vare
    invVarEffects = 1 ./  varEffects
    logVarEffects = log.(varEffects)
    nMarkers      = length(α)
    XpRinvX       = XpRinvX
    nblocks       = length(XpRinvX)
    start_pos     = 0
    for i in 1:nblocks
        XpRinvycorr = XRinvArray[i]*yCorr
        αold        = copy(α)
        block_size  = size(XpRinvX[i],1)
        nreps       = block_size #user-defined nreps=block_size, outer_niter =niter/block_size
        for reps = 1:nreps
            for j=1:block_size #additional code to save all sampled αs is needed
                locus_j    = start_pos+j
                rhs        = (XpRinvycorr[j] + xpRinvx[locus_j]*α[locus_j])*invVarRes
                lhs        = xpRinvx[locus_j]*invVarRes + invVarEffects[locus_j]
                invLhs     = 1/lhs
                gHat       = rhs*invLhs
                logDelta1  = -0.5*(log(lhs) + logVarEffects[locus_j] - gHat*rhs) + logPiComp
                probDelta1 = 1/(1+ exp(logDelta0 - logDelta1))
                oldAlpha   = α[locus_j]

                if rand(rng) < probDelta1
                    δ[locus_j] = 1
                    β[locus_j] = gHat + randn(rng)*sqrt(invLhs)
                    α[locus_j] = β[locus_j]
                    BLAS.axpy!(oldAlpha-α[locus_j],view(XpRinvX[i],:,j),XpRinvycorr)
                else
                    if oldAlpha != 0
                        BLAS.axpy!(oldAlpha,view(XpRinvX[i],:,j),XpRinvycorr)
                    end
                    δ[locus_j] = 0
                    β[locus_j] = randn(rng)*sqrt(varEffects[locus_j])
                    α[locus_j] = 0
                end
            end
        end

        yCorr[:]  = yCorr + XArray[i]*(αold-α)[(start_pos+1):(start_pos+block_size)] #?subset at first is better
        start_pos += block_size
    end
end
