function megaBayesL!(genotypes, wArray, vare; rngs=nothing)
    Threads.@threads for i in 1:length(wArray) #ntraits
        tid = Threads.threadid()
        rng = (rngs === nothing || length(rngs) < tid) ? Random.default_rng() : rngs[tid]
        BayesL!(genotypes.mArray,genotypes.mRinvArray,genotypes.mpRinvm,
            wArray[i],genotypes.α[i],genotypes.gammaArray,vare[i,i],genotypes.G.val[i,i]; rng=rng)
    end
end

function BayesL!(genotypes, ycorr, vare; rng=Random.default_rng())
    BayesL!(genotypes.mArray,genotypes.mRinvArray,genotypes.mpRinvm,
            ycorr,genotypes.α[1],genotypes.gammaArray,vare,genotypes.G.val; rng=rng)
end

function megaBayesC0!(genotypes, wArray, vare; rngs=nothing)
    Threads.@threads for i in 1:length(wArray) #ntraits
        tid = Threads.threadid()
        rng = (rngs === nothing || length(rngs) < tid) ? Random.default_rng() : rngs[tid]
        BayesL!(genotypes.mArray,genotypes.mRinvArray,genotypes.mpRinvm,
                wArray[i],genotypes.α[i],[1.0],vare[i,i],genotypes.G.val[i,i]; rng=rng)
    end
end

function BayesC0!(genotypes, ycorr, vare; rng=Random.default_rng())
    BayesL!(genotypes.mArray,genotypes.mRinvArray,genotypes.mpRinvm,
            ycorr,genotypes.α[1],[1.0],vare,genotypes.G.val; rng=rng)
end

function BayesL!(xArray,xRinvArray,xpRinvx,
                 yCorr,
                 α,gammaArray,
                 vRes,vEff; rng=Random.default_rng())
    nMarkers = length(α)
    λ        = vRes/vEff
    function get_lambda_function(x)
        f1(x,j,λ)  = λ/x[j]
        f2(x,j,λ)  = λ
        length(x)>1 ? f1 : f2
    end
    getlambda = get_lambda_function(gammaArray)
    for j=1:nMarkers
        x, xRinv = xArray[j], xRinvArray[j]
        rhs      = dot(xRinv,yCorr) + xpRinvx[j]*α[j]
        lhs      = xpRinvx[j] + getlambda(gammaArray,j,λ)
        invLhs   = 1.0/lhs
        mean     = invLhs*rhs
        oldAlpha = α[j]
        α[j]     = mean + randn(rng)*sqrt(invLhs*vRes)
        BLAS.axpy!(oldAlpha-α[j],x,yCorr)
    end
end
