#=
================================================================================
MCMC Sampler for NNMM - Bayesian Alphabet Methods
================================================================================
Implements the main MCMC loop for the Neural Network Mixed Model.

Supports: RR-BLUP, GBLUP, BayesA, BayesB, BayesC, BayesL

The sampler alternates between:
  1. Layer 1→2: Sample marker effects (genotypes → omics)
  2. Layer 2→3: Sample omics effects (omics → phenotypes)
  3. Sample latent/missing omics values via HMC or Metropolis-Hastings

Key Variables:
  - mme1: Mixed Model Equations for layer 1→2
  - mme2: Mixed Model Equations for layer 2→3
  - ycorr1: Residuals for layer 1→2
  - ycorr2: Residuals for layer 2→3

Author: NNMM.jl Team
================================================================================
=#

"""
    nnmm_MCMC_BayesianAlphabet(mme1, df1, mme2, df2)

Internal function: Run the MCMC sampler for NNMM.

This is called by `runNNMM` after model setup is complete.

# Arguments
- `mme1`: MME object for genotypes → omics layer
- `df1`: DataFrame containing omics data
- `mme2`: MME object for omics → phenotypes layer  
- `df2`: DataFrame containing phenotype data

# Returns
Dictionary with posterior results including:
- Location parameters
- Variance components
- Marker effects
- EBV estimates
"""
function nnmm_MCMC_BayesianAlphabet(mme1,df1,mme2,df2)
    
    #1->2:
    ############################################################################
    chain_length             = mme1.MCMCinfo.chain_length
    burnin                   = mme1.MCMCinfo.burnin
    output_samples_frequency = mme1.MCMCinfo.output_samples_frequency
    output_prediction_frequency = mme1.MCMCinfo.output_prediction_frequency
    output_folder            = mme1.MCMCinfo.output_folder
    
    invweights1               = mme1.invweights
    update_priors_frequency1  = mme1.MCMCinfo.update_priors_frequency
    missing_phenotypes1       = mme1.MCMCinfo.missing_phenotypes
    is_multi_trait1           = mme1.nModels != 1
    is_nnbayes_partial       = mme1.nonlinear_function != false && mme1.is_fully_connected==false
    is_activation_fcn        = mme1.is_activation_fcn
    nonlinear_function       = mme1.nonlinear_function
    is_linear_activation     = false
    if is_activation_fcn == true && nonlinear_function != false
        is_linear_activation = occursin("mylinear", string(nameof(typeof(nonlinear_function))))
    end
    causal_structure         = false
    debug_scale              = get(ENV, "NNMM_DEBUG_SCALE", "0") == "1"
    debug_scale_iters_str    = get(ENV, "NNMM_DEBUG_SCALE_ITERS", "5")
    debug_scale_iters_parsed = tryparse(Int, debug_scale_iters_str)
    debug_scale_iters        = debug_scale_iters_parsed === nothing ? 5 : debug_scale_iters_parsed
    debug_invariants              = get(ENV, "NNMM_DEBUG_INVARIANTS", "0") == "1"
    debug_invariants_iters_str    = get(ENV, "NNMM_DEBUG_INVARIANTS_ITERS", "5")
    debug_invariants_iters_parsed = tryparse(Int, debug_invariants_iters_str)
    debug_invariants_iters        = debug_invariants_iters_parsed === nothing ? 5 : debug_invariants_iters_parsed

    # Deterministic per-thread RNGs (used by threaded marker samplers) when a seed is provided.
    thread_rngs = nothing
    if mme1.MCMCinfo.seed !== false
        thread_rngs = [Random.MersenneTwister(mme1.MCMCinfo.seed + tid) for tid in 1:Threads.nthreads()]
    end

    # Exclude "prediction-only" individuals from ALL parameter-updating steps.
    #
    # Definition (edge-case guard):
    # - phenotype (yobs) is missing, AND
    # - ALL hidden-layer nodes (omics / latent traits) were unobserved in the input data.
    #
    # Such individuals carry no likelihood information for 1->2 or 2->3 updates; including them can
    # destabilize variance updates via degenerate residuals. We keep them for prediction, but set
    # their weights to zero so they do not contribute to parameter updates.
    if mme1.nonlinear_function != false && mme1.latent_traits != false
        nobs = length(invweights1)
        if length(mme1.yobs) == nobs && (mme1.missingPattern isa AbstractArray) && size(mme1.missingPattern, 1) == nobs
            has_yobs = .!ismissing.(mme1.yobs)
            has_any_hidden_obs = vec(sum(mme1.missingPattern, dims=2) .> 0)
            prediction_only = (.!has_yobs) .& (.!has_any_hidden_obs)
            n_excluded = count(prediction_only)
            if n_excluded > 0
                invweights1[prediction_only] .= zero(eltype(invweights1))
                example_idx = findall(prediction_only)
                example_ids = mme1.obsID[example_idx[1:min(end, 8)]]
                printstyled(
                    "NNMM: excluded $n_excluded individual(s) from parameter updates because they have missing phenotype ($(mme1.yobs_name)) and no observed hidden-layer data. These individuals will be treated as prediction-only.\n" *
                    "      Example IDs: " * join(example_ids, ", ") * "\n",
                    bold=false,
                    color=:yellow,
                )
            end
        else
            @warn "NNMM prediction-only exclusion skipped due to unexpected dimensions" nobs=length(invweights1) yobs_len=length(mme1.yobs) missingPattern_type=typeof(mme1.missingPattern) missingPattern_size=(mme1.missingPattern isa AbstractArray ? size(mme1.missingPattern) : nothing)
        end
    end

    # Helper: get the marker design matrix for layer 2→3 marker classes.
    # - Omics: uses activation-transformed omics aligned to phenotypes
    # - Genotypes: uses aligned genotype covariate matrix (skip connection)
    marker_matrix_23(Mi) = Mi isa Omics ? Mi.aligned_omics_w_phenotype :
                           (Mi isa Genotypes ? Mi.genotypes :
                            error("NNMM: unsupported marker class in 2->3: $(typeof(Mi))"))

    middle_trait_index = Dict(string(mme1.lhsVec[i]) => i for i in 1:mme1.nModels)

    function omics_classes_23()
        mme2.M == 0 ? Omics[] : [Mi for Mi in mme2.M if Mi isa Omics]
    end

    function middle_weights_23()
        T = Float64
        for Mi in omics_classes_23()
            if Mi.α != false && !isempty(Mi.α)
                T = eltype(Mi.α[1])
                break
            end
        end
        weights = zeros(T, mme1.nModels)
        for Mi in omics_classes_23()
            for (j, fid) in enumerate(string.(Mi.featureID))
                if !haskey(middle_trait_index, fid)
                    error("NNMM: omics class '$(Mi.name)' includes '$fid', but it is not a latent trait in 1->2.")
                end
                weights[middle_trait_index[fid]] = Mi.α[1][j]
            end
        end
        return weights
    end

    function middle_weight_variances_23()
        variances = zeros(Float64, mme1.nModels)
        for Mi in omics_classes_23()
            for (j, fid) in enumerate(string.(Mi.featureID))
                if haskey(middle_trait_index, fid)
                    val = Mi.G.val
                    variances[middle_trait_index[fid]] = val isa Number ? Float64(val) : Float64(val[j, j])
                end
            end
        end
        return variances
    end

    function assign_latent_traits_to_omics_classes!(ylats)
        for Mi in omics_classes_23()
            colidx = [middle_trait_index[string(fid)] for fid in Mi.featureID]
            Mi.data[!, Mi.featureID] = ylats[:, colidx]
        end
        return nothing
    end

    function refresh_omics_gibbs_mats_23!(invweights)
        for Mi in omics_classes_23()
            Mi_genotypes = convert(mme2.MCMCinfo.double_precision ? Matrix{Float64} : Matrix{Float32},
                                   marker_matrix_23(Mi))
            mGibbs = GibbsMats(Mi_genotypes, invweights)
            Mi.mArray, Mi.mRinvArray, Mi.mpRinvm = mGibbs.xArray, mGibbs.xRinvArray, mGibbs.xpRinvx
        end
        return nothing
    end

    function middle_matrix_phenotype_order_23()
        omics_classes = omics_classes_23()
        isempty(omics_classes) && error("NNMM: no omics classes found in 2->3.")
        base = omics_classes[1].aligned_omics_w_phenotype
        X = zeros(eltype(base), size(base, 1), mme1.nModels)
        for Mi in omics_classes
            Xi = Mi.aligned_omics_w_phenotype
            for (j, fid) in enumerate(string.(Mi.featureID))
                if haskey(middle_trait_index, fid)
                    X[:, middle_trait_index[fid]] = Xi[:, j]
                end
            end
        end
        return X
    end

    ############################################################################
    # Working Variables
    # 1) samples at current iteration (starting values default to zeros)
    # 2) posterior mean and variance at current iteration (zeros at the beginning)
    # 3) ycorr: phenotypes corrected for all effects
    ############################################################################
    #location parameters
    #mme.sol (its starting values were set in runMCMC)
    mme1.solMean, mme1.solMean2  = zero(mme1.sol),zero(mme1.sol)
    #residual variance
    mme1.meanVare  = zero(mme1.R.val)
    mme1.meanVare2 = zero(mme1.R.val)

    #polygenic effects (pedigree), e.g, Animal+ Maternal
    if mme1.pedTrmVec != 0
       mme1.G0Mean,mme1.G0Mean2  = zero(mme1.Gi),zero(mme1.Gi)
    end
    #marker effects
    if mme1.M != 0
        for Mi in mme1.M
            #Mi.α  (starting values were set in get_genotypes)
            mGibbs    = GibbsMats(Mi.genotypes,invweights1)
            Mi.mArray,Mi.mRinvArray,Mi.mpRinvm  = mGibbs.xArray,mGibbs.xRinvArray,mGibbs.xpRinvx

            if Mi.method=="BayesB" #α=β.*δ
                Mi.G.val        = fill(Mi.G.val,Mi.nMarkers) #a scalar in BayesC but a vector in BayeB
            end
            if Mi.method=="BayesL"         #in the MTBayesLasso paper
                if mme1.nModels == 1
                    Mi.G.val   /= 8           #mme.M.G.val is the scale Matrix, Sigma
                    Mi.G.scale /= 8
                    gammaDist  = Gamma(1, 8) #8 is the scale parameter of the Gamma distribution (1/8 is the rate parameter)
                else
                    Mi.G.val         /= 4*(Mi.ntraits+1)
                    Mi.G.scale     /= 4*(Mi.ntraits+1)
                    gammaDist     = Gamma((Mi.ntraits+1)/2, 8) #8 (1/8): the scale (rate) parameter
                end
                Mi.gammaArray = rand(gammaDist,Mi.nMarkers)
            end
            if Mi.method=="GBLUP"
                GBLUP_setup(Mi)
            end
            Mi.β                  = [copy(Mi.α[traiti]) for traiti = 1:Mi.ntraits] #partial marker effeccts used in BayesB
            Mi.δ                  = [ones(typeof(Mi.α[traiti][1]),Mi.nMarkers) for traiti = 1:Mi.ntraits] #inclusion indicator for marker effects
            Mi.meanAlpha          = [zero(Mi.α[traiti]) for traiti = 1:Mi.ntraits] #marker effects
            Mi.meanAlpha2         = [zero(Mi.α[traiti]) for traiti = 1:Mi.ntraits] #marker effects
            Mi.meanDelta          = [zero(Mi.δ[traiti]) for traiti = 1:Mi.ntraits] #inclusion indicator for marker effects
            Mi.meanVara           = zero(mme1.R.val)  #posterir mean of variance for marker effect
            Mi.meanVara2          = zero(mme1.R.val)  #variable to save variance for marker effect
            Mi.meanScaleVara      = zero(mme1.R.val) #variable to save Scale parameter for prior of marker effect variance
            Mi.meanScaleVara2     = zero(mme1.R.val)  #variable to save Scale parameter for prior of marker effect variance
            if is_multi_trait1
                # if is_mega_trait
                if Mi.G.constraint==true
                    Mi.π        = zeros(Mi.ntraits)
                    Mi.mean_pi  = zeros(Mi.ntraits)
                    Mi.mean_pi2 = zeros(Mi.ntraits)
                else
                    Mi.π,Mi.mean_pi,Mi.mean_pi2 = copy(Mi.π),copy(Mi.π),copy(Mi.π)
                    if Mi.estimatePi == true
                      for key in keys(Mi.mean_pi)
                        Mi.mean_pi[key]=0.0
                        Mi.mean_pi2[key]=0.0
                      end
                    end
                    #if methods == "BayesCC"  labels,BigPi,BigPiMean=setPi(Pi)
                end
            else
                Mi.mean_pi,Mi.mean_pi2 = 0.0,0.0      #inclusion probability
            end
        end
    end
    if is_nnbayes_partial
        nnbayes_partial_para_modify3(mme1)
    end

    #phenotypes corrected for all effects
    ycorr1 = vec(Matrix(mme1.ySparse)-mme1.X*mme1.sol)
    if mme1.M != 0
        for Mi in mme1.M
            for traiti in 1:Mi.ntraits
                if Mi.α[traiti] != zero(Mi.α[traiti])
                    ycorr1[(traiti-1)*Mi.nObs+1 : traiti*Mi.nObs] = ycorr1[(traiti-1)*Mi.nObs+1 : traiti*Mi.nObs]
                                                                 - Mi.genotypes*Mi.α[traiti]
                end
            end
        end
    end
    ############################################################################
    #More on Multi-Trait
    ############################################################################
    if is_multi_trait1
        wArray1 = Array{Union{Array{Float64,1},Array{Float32,1}}}(undef,mme1.nModels)
        for traiti = 1:mme1.nModels
            startPosi             = (traiti-1)*length(mme1.obsID)  + 1
            ptr                   = pointer(ycorr1,startPosi)
            wArray1[traiti]        = unsafe_wrap(Array,ptr,length(mme1.obsID))
        end

        #Starting value for Ri is made based on missing value pattern
        #(imputed phenotypes will not used to compute first mmeRhs)
        Ri         = mkRi(mme1,df1,invweights1)
        dropzeros!(Ri)
    end
    ############################################################################
    #  SET UP OUTPUT MCMC samples
    ############################################################################
    if output_samples_frequency != 0
        outfile=output_MCMC_samples_setup(mme1,chain_length-burnin,
                                          output_samples_frequency,
                                          output_folder*"/MCMC_samples")
        # Set up Layer 2 output files for residual and effect variances
        outfile["layer2_residual_variance"] = open(output_folder*"/MCMC_samples_layer2_residual_variance.txt","w")
        outfile["layer2_effect_variance"] = open(output_folder*"/MCMC_samples_layer2_effect_variance.txt","w")
        println("The file "*output_folder*"/MCMC_samples_layer2_residual_variance.txt is created to save MCMC samples for layer2_residual_variance.")
        println("The file "*output_folder*"/MCMC_samples_layer2_effect_variance.txt is created to save MCMC samples for layer2_effect_variance.")
        # Set up EPV (Estimated Phenotypic Value) output file
        # EPV uses OBSERVED omics instead of predicted omics (EBV uses predicted)
        if nonlinear_function != false
            outfile["EPV_NonLinear"] = open(output_folder*"/MCMC_samples_EPV_NonLinear.txt","w")
            println("The file "*output_folder*"/MCMC_samples_EPV_NonLinear.txt is created to save MCMC samples for EPV_NonLinear.")
            # EPV on output IDs (e.g., test individuals whose phenotypes are missing)
            if mme1.output_ID != 0
                outfile["EPV_Output_NonLinear"] = open(output_folder*"/MCMC_samples_EPV_Output_NonLinear.txt","w")
                println("The file "*output_folder*"/MCMC_samples_EPV_Output_NonLinear.txt is created to save MCMC samples for EPV_Output_NonLinear.")
            end
        end
    end
    ############################################################################
    # MCMC (starting values for sol (zeros);  mme.RNew; G0 are used)
    ############################################################################
    # # Initialize mme for hmc before Gibbs
    if nonlinear_function != false
        # mme1.weights_NN    = vcat(mean(mme1.ySparse),zeros(mme1.nModels))
        mme1.weights_NN    = zeros(mme1.nModels) # intercept is not included
    end
    if mme1.pedTrmVec!=0
        polygenic_pos = findfirst(i -> i.randomType=="A", mme1.rndTrmVec)
    end




    #2->3:
    ############################################################################
    invweights2               = copy(mme2.invweights)  # copy to avoid modifying original
    update_priors_frequency2  = mme2.MCMCinfo.update_priors_frequency
    missing_phenotypes2       = mme2.MCMCinfo.missing_phenotypes
    is_multi_trait2           = mme2.nModels != 1
    # has_categorical_trait    = "categorical"         ∈ mme.traits_type
    # has_binary_trait         = "categorical(binary)" ∈ mme.traits_type
    # has_censored_trait       = "censored"            ∈ mme.traits_type
    has_categorical_trait = false
    has_binary_trait = false
    has_censored_trait = false

    # Exclude individuals with missing phenotypes from Layer 2→3 variance estimation.
    # These individuals cannot contribute to residual variance estimation because their
    # phenotype is unknown. Similar to JWAS's trainingInd filtering approach.
    # Note: mme2.yobs contains the phenotype data aligned with mme2.obsID.
    if mme2.ySparse !== nothing && invweights2 != false
        # mme2.ySparse contains phenotypes (with missing replaced by 0), check the original yobs
        # which is stored in mme1.yobs but indexed by mme1.obsID. We need to use mme2's data.
        # For Layer 2→3, we check if the phenotype was originally missing.
        # The missingPattern or yobs in mme1 tracks this, but dimensions may differ.
        # Use the prediction_only mask which was already computed for Layer 1.
        if mme1.nonlinear_function != false && mme1.latent_traits != false
            nobs2 = length(invweights2)
            if length(mme1.yobs) == nobs2
                missing_pheno = ismissing.(mme1.yobs)
                if any(missing_pheno)
                    invweights2[missing_pheno] .= zero(eltype(invweights2))
                end
            end
        end
    end

    ############################################################################
    # Working Variables
    # 1) samples at current iteration (starting values default to zeros)
    # 2) posterior mean and variance at current iteration (zeros at the beginning)
    # 3) ycorr: phenotypes corrected for all effects
    ############################################################################
    #location parameters
    #mme.sol (its starting values were set in runMCMC)
    mme2.solMean, mme2.solMean2  = zero(mme2.sol),zero(mme2.sol)
    #residual variance
    mme2.meanVare  = zero(mme2.R.val)
    mme2.meanVare2 = zero(mme2.R.val)

    #polygenic effects (pedigree), e.g, Animal+ Maternal
    if mme2.pedTrmVec != 0
       mme2.G0Mean,mme2.G0Mean2  = zero(mme2.Gi),zero(mme2.Gi)
    end
    #marker effects
    if mme2.M != 0
        for Mi in mme2.M
            #Mi.α  (starting values were set in get_genotypes)
            Mi_genotypes = convert(mme2.MCMCinfo.double_precision ? Matrix{Float64} : Matrix{Float32},
                                   marker_matrix_23(Mi))
            mGibbs    = GibbsMats(Mi_genotypes,invweights2)
            Mi.mArray,Mi.mRinvArray,Mi.mpRinvm  = mGibbs.xArray,mGibbs.xRinvArray,mGibbs.xpRinvx

            if Mi.method=="BayesB" #α=β.*δ
                Mi.G.val        = fill(Mi.G.val,Mi.nMarkers) #a scalar in BayesC but a vector in BayeB
            end
            if Mi.method=="BayesL"         #in the MTBayesLasso paper
                if mme2.nModels == 1
                    Mi.G.val   /= 8           #mme.M.G.val is the scale Matrix, Sigma
                    Mi.G.scale /= 8
                    gammaDist  = Gamma(1, 8) #8 is the scale parameter of the Gamma distribution (1/8 is the rate parameter)
                else
                    Mi.G.val         /= 4*(Mi.ntraits+1)
                    Mi.G.scale     /= 4*(Mi.ntraits+1)
                    gammaDist     = Gamma((Mi.ntraits+1)/2, 8) #8 (1/8): the scale (rate) parameter
                end
                Mi.gammaArray = rand(gammaDist,Mi.nMarkers)
            end
            if Mi.method=="GBLUP"
                GBLUP_setup(Mi)
            end
            Mi.β                  = [copy(Mi.α[traiti]) for traiti = 1:Mi.ntraits] #partial marker effeccts used in BayesB
            Mi.δ                  = [ones(typeof(Mi.α[traiti][1]),Mi.nMarkers) for traiti = 1:Mi.ntraits] #inclusion indicator for marker effects
            Mi.meanAlpha          = [zero(Mi.α[traiti]) for traiti = 1:Mi.ntraits] #marker effects
            Mi.meanAlpha2         = [zero(Mi.α[traiti]) for traiti = 1:Mi.ntraits] #marker effects
            Mi.meanDelta          = [zero(Mi.δ[traiti]) for traiti = 1:Mi.ntraits] #inclusion indicator for marker effects
            Mi.meanVara           = zero(mme2.R.val)  #posterir mean of variance for marker effect
            Mi.meanVara2          = zero(mme2.R.val)  #variable to save variance for marker effect
            Mi.meanScaleVara      = zero(mme2.R.val) #variable to save Scale parameter for prior of marker effect variance
            Mi.meanScaleVara2     = zero(mme2.R.val)  #variable to save Scale parameter for prior of marker effect variance
            if is_multi_trait2
                # if is_mega_trait
                if Mi.G.constraint==true
                    Mi.π        = zeros(Mi.ntraits)
                    Mi.mean_pi  = zeros(Mi.ntraits)
                    Mi.mean_pi2 = zeros(Mi.ntraits)
                else
                    Mi.π,Mi.mean_pi,Mi.mean_pi2 = copy(Mi.π),copy(Mi.π),copy(Mi.π)
                    if Mi.estimatePi == true
                      for key in keys(Mi.mean_pi)
                        Mi.mean_pi[key]=0.0
                        Mi.mean_pi2[key]=0.0
                      end
                    end
                    #if methods == "BayesCC"  labels,BigPi,BigPiMean=setPi(Pi)
                end
            else
                Mi.mean_pi,Mi.mean_pi2 = 0.0,0.0      #inclusion probability
            end
        end
    end

    # phenotypes corrected for all effects (2->3)
    # NOTE: ycorr2 must always be y - X*b - Σ(Z_i * α_i) before calling BayesABC!,
    # otherwise α will drift (double-counting the previous iteration's effects).
    ycorr2 = vec(Matrix(mme2.ySparse) - mme2.X * mme2.sol) #length of ycorr2 is #individuals with non-missing yobs
    if mme2.M != 0
        for Mi in mme2.M
            # MT 2->3 is not supported for now, so Mi.ntraits is expected to be 1
            ycorr2 .-= marker_matrix_23(Mi) * Mi.α[1]
        end
    end

    ############################################################################
    # Initialize wArray2 (always, for both single-trait and multi-trait)
    # wArray2[i] is a view into ycorr2 for trait i
    # For single-trait: wArray2[1] ≡ ycorr2
    ############################################################################
    wArray2 = Array{Union{Array{Float64,1},Array{Float32,1}}}(undef,mme2.nModels)
    for traiti = 1:mme2.nModels
        startPosi             = (traiti-1)*length(mme2.obsID)  + 1
        ptr                   = pointer(ycorr2,startPosi)
        wArray2[traiti]        = unsafe_wrap(Array,ptr,length(mme2.obsID))
    end

    if is_multi_trait2
        #Starting value for Ri is made based on missing value pattern
        #(imputed phenotypes will not used to compute first mmeRhs)
        Ri         = mkRi(mme2,df2,invweights2)
        dropzeros!(Ri)
    end
    ############################################################################
    #  SET UP OUTPUT MCMC samples
    ############################################################################
    # if output_samples_frequency != 0
    #     outfile=output_MCMC_samples_setup(mme,chain_length-burnin,
    #                                       output_samples_frequency,
    #                                       output_folder*"/MCMC_samples")
    # end
    ############################################################################
    # MCMC (starting values for sol (zeros);  mme.RNew; G0 are used)
    ############################################################################

    if mme2.pedTrmVec!=0
        polygenic_pos = findfirst(i -> i.randomType=="A", mme2.rndTrmVec)
    end

    # Write header for EPV_NonLinear file (IDs of phenotyped individuals)
    if output_samples_frequency != 0 && nonlinear_function != false && mme2.M != 0 && length(mme2.M) > 0
        epv_ids = omics_classes_23()[1].aligned_obsID_w_phenotype
        writedlm(outfile["EPV_NonLinear"], transubstrarr(epv_ids), ',')
        if mme1.output_ID != 0 && haskey(outfile, "EPV_Output_NonLinear")
            writedlm(outfile["EPV_Output_NonLinear"], transubstrarr(mme1.output_ID), ',')
        end
    end

    # Precompute sparse incidence matrices used repeatedly inside the MCMC loop.
    # These only depend on IDs (which are fixed during MCMC) and help avoid
    # rebuilding and re-allocating sparse matrices each iteration.
    Z_yobs_from_omics = nothing
    if nonlinear_function != false && mme2.M != 0 && length(mme2.M) > 0
        first_omics = omics_classes_23()[1]
        Z_yobs_from_omics = mkmat_incidence_factor(mme2.obsID, first_omics.obsID)
        Z_yobs_from_omics = map(mme1.MCMCinfo.double_precision ? Float64 : Float32, Z_yobs_from_omics)
    end

    Z_obs_from_layer2_omics = nothing
    if Z_yobs_from_omics !== nothing && (mme1.obsID != omics_classes_23()[1].obsID)
        Z_obs_from_layer2_omics = mkmat_incidence_factor(mme1.obsID, omics_classes_23()[1].obsID)
        Z_obs_from_layer2_omics = map(mme1.MCMCinfo.double_precision ? Float64 : Float32, Z_obs_from_layer2_omics)
    end

    Z_output_from_obs = nothing
    if mme1.output_ID != 0 && mme1.output_ID != mme1.obsID
        Z_output_from_obs = mkmat_incidence_factor(mme1.output_ID, mme1.obsID)
        Z_output_from_obs = map(mme1.MCMCinfo.double_precision ? Float64 : Float32, Z_output_from_obs)
    end

    @showprogress "running MCMC ..." for iter=1:chain_length
        
        #1->2
        ########################################################################
        # 1. Non-Marker Location Parameters
        ########################################################################
        # 1.1 Update Left-hand-side of MME
        if is_multi_trait1
            mme1.mmeLhs = mme1.X'Ri*mme1.X #normal equation, Ri is changed
            dropzeros!(mme1.mmeLhs)
        end
        addVinv(mme1)
        # 1.2 Update Right-hand-side of MME
        if is_multi_trait1
            if mme1.MCMCinfo.missing_phenotypes==true
              ycorr1[:]=sampleMissingResiduals(mme1,ycorr1)
            end
        end
        ycorr1[:] = ycorr1 + mme1.X*mme1.sol
        if is_multi_trait1
            mme1.mmeRhs =  mme1.X'Ri*ycorr1
        else
            mme1.mmeRhs = (invweights1 == false) ? mme1.X'ycorr1 : mme1.X'Diagonal(invweights1)*ycorr1
        end
        # 1.3 Gibbs sampler
        if is_multi_trait1
            Gibbs(mme1.mmeLhs,mme1.sol,mme1.mmeRhs)
        else
            Gibbs(mme1.mmeLhs,mme1.sol,mme1.mmeRhs,mme1.R.val)
        end

        ycorr1[:] = ycorr1 - mme1.X*mme1.sol
        ########################################################################
        # 2. Marker Effects
        ########################################################################
	        if mme1.M !=0
	            for i in 1:length(mme1.M)
	                Mi=mme1.M[i]
                ########################################################################
                # Marker Effects
                ########################################################################
                if Mi.method in ["BayesC","BayesB","BayesA"]
                    locus_effect_variances = (Mi.method == "BayesC" ? fill(Mi.G.val,Mi.nMarkers) : Mi.G.val)
	                    if is_multi_trait1 && !is_nnbayes_partial
	                        if Mi.G.constraint==true
	                            megaBayesABC!(Mi, wArray1, mme1.R.val, locus_effect_variances; rngs=thread_rngs)
	                        else
	                            MTBayesABC!(Mi,wArray1,mme1.R.val,locus_effect_variances,mme1.nModels)
	                        end
                    elseif is_nnbayes_partial
                        BayesABC!(Mi,wArray1[i],mme1.R.val[i,i],locus_effect_variances) #this can be parallelized (conflict with others)
                    else
                        BayesABC!(Mi,ycorr1,mme1.R.val,locus_effect_variances)
                    end
                elseif Mi.method =="RR-BLUP"
	                    if is_multi_trait1 && !is_nnbayes_partial
	                        if Mi.G.constraint==true
	                            megaBayesC0!(Mi, wArray1, mme1.R.val; rngs=thread_rngs)
	                        else
	                            MTBayesC0!(Mi,wArray1,mme1.R.val)
	                        end
                    elseif is_nnbayes_partial
                        BayesC0!(Mi,wArray1[i],mme1.R.val[i,i])
                    else
                        BayesC0!(Mi,ycorr1,mme1.R.val)
                    end
                elseif Mi.method == "BayesL"
	                    if is_multi_trait1 && !is_nnbayes_partial
	                        #problem with sampleGammaArray
	                        if Mi.G.constraint==true
	                            megaBayesL!(Mi, wArray1, mme1.R.val; rngs=thread_rngs)
	                        else
	                            MTBayesL!(Mi,wArray1,mme1.R.val)
	                        end
                    elseif is_nnbayes_partial
                        BayesC0!(Mi,wArray1[i],mme1.R.val[i,i])
                    else
                        BayesL!(Mi,ycorr1,mme1.R.val)
                    end
                elseif Mi.method == "GBLUP"
	                    if is_multi_trait1 && !is_nnbayes_partial
	                        if Mi.G.constraint==true
	                            megaGBLUP!(Mi, wArray1, mme1.R.val, invweights1; rngs=thread_rngs)
	                        else
	                            MTGBLUP!(Mi,wArray1,ycorr1,mme1.R.val,invweights1)
	                        end
                    elseif is_nnbayes_partial
                        GBLUP!(Mi,wArray1[i],mme1.R.val[i,i],invweights1)
                    else
                        GBLUP!(Mi,ycorr1,mme1.R.val,invweights1)
                    end
                end
                ########################################################################
                # Marker Inclusion Probability
                ########################################################################
                if Mi.estimatePi == true
                    if is_multi_trait1 && !is_nnbayes_partial
                        if Mi.G.constraint==true
                            Mi.π = [samplePi(sum(Mi.δ[i]), Mi.nMarkers) for i in 1:mme1.nModels]
                        else
                            samplePi(Mi.δ,Mi.π) #samplePi(deltaArray,Mi.π,labels)
                        end
                    else
                        Mi.π = samplePi(sum(Mi.δ[1]), Mi.nMarkers)
                    end
                end
                ########################################################################
                # Variance of Marker Effects
                ########################################################################
                if Mi.G.estimate_variance == true #methd specific estimate_variance
                    sample_marker_effect_variance(Mi)
                    if mme1.MCMCinfo.double_precision == false && Mi.method != "BayesB"
                        Mi.G.val = Float32.(Mi.G.val)
                    end
                end
                ########################################################################
                # Scale Parameter in Priors for Marker Effect Variances
                ########################################################################
                if Mi.G.estimate_scale == true
                    if !is_multi_trait1
                        a = size(Mi.G.val,1)*Mi.G.df/2   + 1
                        b = sum(Mi.G.df ./ (2*Mi.G.val)) + 1
                        Mi.G.scale = rand(Gamma(a,1/b))
                    end
	                end
	            end
	        end
		        if debug_invariants && iter <= debug_invariants_iters && !is_nnbayes_partial
		            # Expensive but robust check: recompute residuals from scratch and compare to
		            # the in-place updated `ycorr1`. Useful to catch silent drift bugs.
		            ycorr1_check = vec(mme1.ySparse) - mme1.X * mme1.sol
		            if mme1.M != 0
		                for Mi in mme1.M
		                    for traiti in 1:Mi.ntraits
		                        ycorr1_check[(traiti-1)*Mi.nObs+1 : traiti*Mi.nObs] .-= Mi.genotypes * Mi.α[traiti]
		                    end
		                end
		            end
		            println("[NNMM_DEBUG_INVARIANTS iter=$(iter)] ycorr1 maxabs(check-current)=$(maximum(abs.(ycorr1_check .- ycorr1)))")
		        end
	        ########################################################################
	        # 3. Non-marker Variance Components
	        ########################################################################

        ########################################################################
        # 3.1 Variance of Non-marker Random Effects
        # e.g, i.i.d; polygenic effects (pedigree)
        ########################################################################
        if length(mme1.rndTrmVec)>0
            if mme1.rndTrmVec[1].Gi.estimate_variance == true
                sampleVCs(mme1,mme1.sol)
            end
        end
        ########################################################################
        # 3.2 Residual Variance
        ########################################################################
	        if mme1.R.estimate_variance == true
	            has_binary_trait=false #NNMM does not support binary traits now
	            # Use effective sample size: only individuals with invweights > 0 contribute to SSE,
	            # so degrees of freedom should also use this count (fixes bias when prediction-only
	            # individuals are excluded via invweights=0). See JWAS's nTrain approach.
	            n_effective1 = (invweights1 == false) ? length(mme1.obsID) : count(invweights1 .> 0)
	            if is_multi_trait1
	                mme1.R.val = sample_variance(wArray1, n_effective1,
	                                        mme1.R.df, mme1.R.scale,
	                                        invweights1,mme1.R.constraint;
	                                        binary_trait_index=has_binary_trait ? findall(x->x=="categorical(binary)", mme1.traits_type) : false)
	                Ri    = kron(inv(mme1.R.val),spdiagm(0=>invweights1))
	            else #single trait
	                if !has_categorical_trait && !has_binary_trait # fixed =1 for single categorical/binary trait
	                    mme1.ROld  = mme1.R.val
	                    mme1.R.val = sample_variance(ycorr1, n_effective1, mme1.R.df, mme1.R.scale, invweights1)
	                end
	            end
	            if mme1.MCMCinfo.double_precision == false
	                mme1.R.val = Float32.(mme1.R.val)
	            end
	            if mme1.R.val isa Number
	                if !isfinite(mme1.R.val) || mme1.R.val <= 0
	                    error("NNMM: invalid 1->2 residual variance at iter=$iter: $(mme1.R.val)")
	                end
	            else
	                if !all(isfinite, mme1.R.val)
	                    error("NNMM: non-finite 1->2 residual covariance at iter=$iter")
	                end
	            end
	        end

	        ########################################################################
	        # 5. Latent Traits (NNBayes) 
	        # to update ycorr1!
	        # from sample_latent_traits(yobs,mme,ycorr,nonlinear_function)
	        ########################################################################
	        yobs = mme1.yobs
	        ylats_vec = vec(mme1.ySparse)          # current values of each latent trait; stacked by trait
	        # Mean of each latent trait under the 1→2 model:
	        #   y_lats = μ_ylats + e, where e has residuals `ycorr1` and var `mme1.R.val`.
	        # Since `ycorr1` is maintained as the full residual (y_lats - μ_ylats),
	        # we can recover μ_ylats as (y_lats - ycorr1), which includes ALL non-marker
	        # fixed/random terms in `mme1.X * mme1.sol` plus marker terms.
	        μ_ylats_vec = ylats_vec .- ycorr1
	        σ2_yobs      = mme1.σ2_yobs      # residual variance of yobs (scalar)
	        σ2_weightsNN = mme1.σ2_weightsNN # variance of nn weights between middle and output layers
	    
	        #reshape the vector to nind X ntraits
	        nobs, ntraits = length(mme1.obsID), mme1.nModels
	        ylats_old     = reshape(ylats_vec, nobs, ntraits) #Tianjing's mme.Z
	        ylats_old2    = copy(ylats_old) #save original omics data before updating
	        μ_ylats       = reshape(μ_ylats_vec, nobs, ntraits)
	        ycorr_reshape        = reshape(ycorr1,nobs,ntraits)
    
        #sample latent traits (omics) only where omics are incomplete
        incomplete_omics = mme1.incomplete_omics #indicator for ind with no/partial omics
        
        if sum(incomplete_omics) != 0   #at least 1 ind with incomplete omics
            # HMC/MH uses the phenotype likelihood term, so restrict it to inds with observed yobs.
            has_yobs = .!ismissing.(yobs)
            incomplete_with_yobs = incomplete_omics .& has_yobs
            incomplete_no_yobs   = incomplete_omics .& .!has_yobs

	            if any(incomplete_with_yobs)
	                if mme1.is_activation_fcn == true #Neural Network with activation function (incl. linear)
		                    if is_linear_activation
		                        # Linear activation => exact Gaussian conditional for *missing* entries, conditional on observed entries.
		                        #
		                        # Compute y_centered = y - Xb - (genotype-skip effects) in the full middle-layer (mme1) order.
		                        # We subtract genotype-skip terms here so the latent-trait conditional uses only the omics→y mapping.
		                        y_centered_train = vec(Matrix(mme2.ySparse) - mme2.X * mme2.sol)
		                        if mme2.M != 0 && length(mme2.M) > 1
		                            for Mi in mme2.M
		                                if Mi isa Genotypes
		                                    y_centered_train .-= Mi.genotypes * Mi.α[1]
		                                end
		                            end
		                        end
		                        y_centered_all = Z_yobs_from_omics' * y_centered_train
		                        if Z_obs_from_layer2_omics !== nothing
		                            y_centered_all = Z_obs_from_layer2_omics * y_centered_all
		                        end
	                        inc_idx = findall(incomplete_with_yobs)
	                        sample_missing_latent_traits_linear_gaussian!(
	                            view(ylats_old, inc_idx, :),
	                            view(μ_ylats, inc_idx, :),
	                            y_centered_all[inc_idx],
	                            mme1.weights_NN,
	                            mme1.R.val,
	                            σ2_yobs,
	                            mme1.missingPattern[inc_idx, :],
	                        )
	                    else
	                        # Step 1. sample latent traits for incomplete inds with observed yobs via HMC.
	                        #
	                        # Map ycorr2 (2->3 residuals, in mme2.obsID order) back to the
	                        # full omics/individual order so it aligns with `ylats_old` and `yobs`.
	                        ycorr_yobs_all = Z_yobs_from_omics' * ycorr2
	                        if Z_obs_from_layer2_omics !== nothing
	                            ycorr_yobs_all = Z_obs_from_layer2_omics * ycorr_yobs_all
	                        end
	                        ycorr_yobs = ycorr_yobs_all[incomplete_with_yobs]

	                        ylats_new = hmc_one_iteration_masked(10, 0.1,
	                                                            ylats_old[incomplete_with_yobs, :],
	                                                            yobs[incomplete_with_yobs],
	                                                            mme1.weights_NN, mme1.R.val, σ2_yobs,
	                                                            ycorr_reshape[incomplete_with_yobs, :],
	                                                            nonlinear_function,
	                                                            ycorr_yobs,
	                                                            mme1.missingPattern[incomplete_with_yobs, :])

	                        if any(x -> !isfinite(x), ylats_new)
	                            @warn "NNMM: non-finite latent trait draw; keeping previous values for this iteration" iter=iter
	                        else
	                            ylats_old[incomplete_with_yobs, :] = ylats_new
	                        end
	                    end
                else  # user-defined function, MH (phenotype likelihood only)
                    ylats_old_inc = ylats_old[incomplete_with_yobs, :]
                    μ_ylats_inc   = μ_ylats[incomplete_with_yobs, :]
                    yobs_inc      = yobs[incomplete_with_yobs]
                    ninc          = size(ylats_old_inc, 1)

	                    T = eltype(μ_ylats_inc)
	                    # Propose only missing entries; keep observed omics fixed.
	                    observed_mask_inc = mme1.missingPattern[incomplete_with_yobs, :]
	                    missing_mask_inc = .!observed_mask_inc
	                    candidates = copy(ylats_old_inc)
	                    if mme1.R.val isa Number
	                        draws = μ_ylats_inc .+ randn(T, ninc, ntraits) .* sqrt(T(mme1.R.val))
	                    elseif mme1.R.val isa Diagonal
	                        sds = sqrt.(diag(mme1.R.val))
	                        draws = μ_ylats_inc .+ randn(T, ninc, ntraits) .* reshape(T.(sds), 1, :)
	                    else
	                        L = cholesky(Symmetric(mme1.R.val)).L
	                        draws = μ_ylats_inc .+ randn(T, ninc, ntraits) * L'
	                    end
	                    candidates[missing_mask_inc] .= draws[missing_mask_inc]

                    if nonlinear_function == "Neural Network (MH)"
                        error("not supported for now")
                    else # user-defined non-linear function
                        μ_yobs_candidate = nonlinear_function.(Tuple([view(candidates, :, i) for i in 1:ntraits])...)
                        μ_yobs_current   = nonlinear_function.(Tuple([view(ylats_old_inc, :, i) for i in 1:ntraits])...)
                    end
                    llh_current   = -0.5 * (yobs_inc .- μ_yobs_current).^2 / σ2_yobs
                    llh_candidate = -0.5 * (yobs_inc .- μ_yobs_candidate).^2 / σ2_yobs
                    mhRatio       = exp.(llh_candidate - llh_current)
                    updateus      = rand(ninc) .< mhRatio
                    ylats_new     = candidates .* updateus + ylats_old_inc .* (.!updateus)

                    # Update latent traits with sampled values.
                    if any(x -> !isfinite(x), ylats_new)
                        @warn "NNMM: non-finite latent trait draw; keeping previous values for this iteration" iter=iter
                    else
                        ylats_old[incomplete_with_yobs, :] = ylats_new
                    end
                end
	            end

            # For incomplete inds without yobs, sample from the 1->2 conditional normal model only.
            if any(incomplete_no_yobs)
                μ_ylats_inc = μ_ylats[incomplete_no_yobs, :]
                ninc = size(μ_ylats_inc, 1)
                T = eltype(μ_ylats_inc)
                if mme1.R.val isa Number
                    candidates = μ_ylats_inc .+ randn(T, ninc, ntraits) .* sqrt(T(mme1.R.val))
                elseif mme1.R.val isa Diagonal
                    sds = sqrt.(diag(mme1.R.val))
                    candidates = μ_ylats_inc .+ randn(T, ninc, ntraits) .* reshape(T.(sds), 1, :)
                else
                    L = cholesky(Symmetric(mme1.R.val)).L
                    candidates = μ_ylats_inc .+ randn(T, ninc, ntraits) * L'
                end
                if any(x -> !isfinite(x), candidates)
                    error("NNMM: non-finite latent trait candidates for individuals without phenotype at iter=$iter")
                end
                ylats_old[incomplete_no_yobs, :] = candidates
	            end
	        end
        
	        #step 3. for individuals with partial omics data, put back the partial real omics.
	        ylats_old[mme1.missingPattern] .= ylats_old2[mme1.missingPattern]
	        if any(x -> !isfinite(x), ylats_old)
	            error("NNMM: non-finite latent traits after update at iter=$iter")
	        end
    
        #update ylats (i.e., omics)
        mme1.ySparse = vec(ylats_old) #omics data, note that ylats_old is the updated omics data, the old omics data is in ylats_old2

        #update ycorr1 (for 1->2)
        # ycorr1[:]    = mme1.ySparse - vec(μ_ylats) # =(ylats_new - ylats_old) + ycorr: update residuls (ycorr)
        ycorr1[:] += mme1.ySparse - vec(ylats_old2)

        #update data for 2->3
        #update omics data
        assign_latent_traits_to_omics_classes!(ylats_old)
	        #update aligned transformed omics data (g(z))
	        align_transformed_omics_with_phenotypes(mme2,nonlinear_function)
	        #update ycorr2 (2->3): y - X*b - Σ(Z_i * α_i)
	        ycorr2[:] = vec(Matrix(mme2.ySparse) - mme2.X * mme2.sol)
		        if mme2.M != 0
		            for Mi in mme2.M
		                ycorr2 .-= marker_matrix_23(Mi) * Mi.α[1]
		            end
		        end
		        if any(x -> !isfinite(x), ycorr2)
		            n_bad = count(x -> !isfinite(x), ycorr2)
		            error("NNMM: ycorr2 contains $n_bad non-finite value(s) at iter=$iter; aborting to avoid NaN cascades.")
	        end
	        #update Mi.mArray, Mi.mRinvArray, Mi.mpRinvx for 2->3
	        refresh_omics_gibbs_mats_23!(invweights2)
        if debug_scale && iter <= debug_scale_iters && mme2.M != 0 && length(mme2.M) > 0
            Mi_dbg = omics_classes_23()[1]
            X_dbg = Mi_dbg.aligned_omics_w_phenotype
            α_dbg = Mi_dbg.α[1]
            println("\n[NNMM_DEBUG_SCALE iter=$(iter)] pre-2->3:")
            println("  ycorr2: mean=$(mean(ycorr2)) std=$(std(ycorr2)) maxabs=$(maximum(abs.(ycorr2)))")
            println("  X_dbg col1: mean=$(mean(view(X_dbg,:,1))) std=$(std(view(X_dbg,:,1))) maxabs=$(maximum(abs.(view(X_dbg,:,1))))")
            println("  α_dbg: std=$(std(α_dbg)) maxabs=$(maximum(abs.(α_dbg)))")
        end

        #2->3
        ########################################################################
        # 1. Non-Marker Location Parameters
        ########################################################################
        # 1.1 Update Left-hand-side of MME
        if is_multi_trait2
            error("MT 2->3 is not supported for now")
            mme2.mmeLhs = mme2.X'Ri*mme2.X #normal equation, Ri is changed
            dropzeros!(mme2.mmeLhs)
        end
        addVinv(mme2)
        # 1.2 Update Right-hand-side of MME
        if is_multi_trait2
            if mme2.MCMCinfo.missing_phenotypes==true
              ycorr2[:]=sampleMissingResiduals(mme2,ycorr2)
            end
        end
        ycorr2[:] = ycorr2 + mme2.X*mme2.sol
        if is_multi_trait2
            mme2.mmeRhs =  mme2.X'Ri*ycorr2
        else
            mme2.mmeRhs = (invweights2 == false) ? mme2.X'ycorr2 : mme2.X'Diagonal(invweights2)*ycorr2
        end
        # 1.3 Gibbs sampler
        if is_multi_trait2
            Gibbs(mme2.mmeLhs,mme2.sol,mme2.mmeRhs)
        else
            Gibbs(mme2.mmeLhs,mme2.sol,mme2.mmeRhs,mme2.R.val)
        end

        ycorr2[:] = ycorr2 - mme2.X*mme2.sol
        ########################################################################
        # 2. Marker Effects
        ########################################################################
	        if mme2.M !=0
	            for i in 1:length(mme2.M)
	                Mi=mme2.M[i]
                ########################################################################
                # Marker Effects
                ########################################################################
                if Mi.method in ["BayesC","BayesB","BayesA"]
                    locus_effect_variances = (Mi.method == "BayesC" ? fill(Mi.G.val,Mi.nMarkers) : Mi.G.val)
	                    if is_multi_trait2
	                        if Mi.G.constraint==true
	                            megaBayesABC!(Mi, wArray2, mme2.R.val, locus_effect_variances; rngs=thread_rngs)
	                        else
	                            MTBayesABC!(Mi,wArray2,mme2.R.val,locus_effect_variances,mme2.nModels)
	                        end
                    else
                        BayesABC!(Mi,wArray2[1],mme2.R.val,locus_effect_variances)
                    end
                elseif Mi.method =="RR-BLUP"
	                    if is_multi_trait2
	                        if Mi.G.constraint==true
	                            megaBayesC0!(Mi, wArray2, mme2.R.val; rngs=thread_rngs)
	                        else
	                            MTBayesC0!(Mi,wArray2,mme2.R.val)
	                        end
                    else
                        BayesC0!(Mi,wArray2[1],mme2.R.val)
                    end
                elseif Mi.method == "BayesL"
	                    if is_multi_trait2
	                        #problem with sampleGammaArray
	                        if Mi.G.constraint==true
	                            megaBayesL!(Mi, wArray2, mme2.R.val; rngs=thread_rngs)
	                        else
	                            MTBayesL!(Mi,wArray2,mme2.R.val)
	                        end
                    else
                        BayesL!(Mi,wArray2[1],mme2.R.val)
                    end
                elseif Mi.method == "GBLUP"
	                    if is_multi_trait2
	                        if Mi.G.constraint==true
	                            megaGBLUP!(Mi, wArray2, mme2.R.val, invweights2; rngs=thread_rngs)
	                        else
	                            MTGBLUP!(Mi,wArray2,ycorr2,mme2.R.val,invweights2)
	                        end
                    else
                        GBLUP!(Mi,wArray2[1],mme2.R.val,invweights2)
                    end
                end
                if debug_scale && iter <= debug_scale_iters
                    X_dbg = marker_matrix_23(Mi)
                    α_dbg = Mi.α[1]
                    pred_dbg = X_dbg * α_dbg
                    println("[NNMM_DEBUG_SCALE iter=$(iter)] post-2->3 marker effects:")
                    println("  α_dbg: std=$(std(α_dbg)) maxabs=$(maximum(abs.(α_dbg)))")
                    println("  pred=Xα: mean=$(mean(pred_dbg)) std=$(std(pred_dbg)) maxabs=$(maximum(abs.(pred_dbg)))")
                end
                ########################################################################
                # Marker Inclusion Probability
                ########################################################################
                if Mi.estimatePi == true
                    if is_multi_trait2
                        if Mi.G.constraint==true
                            Mi.π = [samplePi(sum(Mi.δ[i]), Mi.nMarkers) for i in 1:mme2.nModels]
                        else
                            samplePi(Mi.δ,Mi.π) #samplePi(deltaArray,Mi.π,labels)
                        end
                    else
                        Mi.π = samplePi(sum(Mi.δ[1]), Mi.nMarkers)
                    end
                end
                ########################################################################
                # Variance of Marker Effects
                ########################################################################
                if Mi.G.estimate_variance == true #methd specific estimate_variance
                    sample_marker_effect_variance(Mi)
                    if mme2.MCMCinfo.double_precision == false && Mi.method != "BayesB"
                        Mi.G.val = Float32.(Mi.G.val)
                    end
                end
                ########################################################################
                # Scale Parameter in Priors for Marker Effect Variances
                ########################################################################
                if Mi.G.estimate_scale == true
                    if !is_multi_trait2
                        a = size(Mi.G.val,1)*Mi.G.df/2   + 1
                        b = sum(Mi.G.df ./ (2*Mi.G.val)) + 1
                        Mi.G.scale = rand(Gamma(a,1/b))
                    end
	                end
	            end
	        end
		        if debug_invariants && iter <= debug_invariants_iters
		            ycorr2_check = vec(mme2.ySparse) - mme2.X * mme2.sol
		            if mme2.M != 0
		                for Mi in mme2.M
		                    ycorr2_check .-= marker_matrix_23(Mi) * Mi.α[1]
		                end
		            end
		            println("[NNMM_DEBUG_INVARIANTS iter=$(iter)] ycorr2 maxabs(check-current)=$(maximum(abs.(ycorr2_check .- ycorr2)))")
		        end
	        ########################################################################
	        # 3. Non-marker Variance Components
	        ########################################################################

        ########################################################################
        # 3.1 Variance of Non-marker Random Effects
        # e.g, i.i.d; polygenic effects (pedigree)
        ########################################################################
        if length(mme2.rndTrmVec)>0
            if mme2.rndTrmVec[1].Gi.estimate_variance == true
                sampleVCs(mme2,mme2.sol)
            end
        end
        ########################################################################
        # 3.2 Residual Variance
        ########################################################################
	        if mme2.R.estimate_variance == true
	            has_binary_trait=false #NNMM does not support binary traits now
	            # Use effective sample size: only individuals with invweights > 0 contribute to SSE,
	            # so degrees of freedom should also use this count. For Layer 2→3, individuals with
	            # missing phenotypes should not contribute to variance estimation.
	            n_effective2 = (invweights2 == false) ? length(mme2.obsID) : count(invweights2 .> 0)
	            if is_multi_trait2
	                mme2.R.val = sample_variance(wArray2, n_effective2,
	                                        mme2.R.df, mme2.R.scale,
	                                        invweights2,mme2.R.constraint;
	                                        binary_trait_index=has_binary_trait ? findall(x->x=="categorical(binary)", mme2.traits_type) : false)
	                Ri    = kron(inv(mme2.R.val),spdiagm(0=>invweights2))
	            else #single trait
	                if !has_categorical_trait && !has_binary_trait # fixed =1 for single categorical/binary trait
	                    mme2.ROld  = mme2.R.val
	                    mme2.R.val = sample_variance(ycorr2, n_effective2, mme2.R.df, mme2.R.scale, invweights2)
	                end
	            end
	            if mme2.MCMCinfo.double_precision == false
	                mme2.R.val = Float32.(mme2.R.val)
	            end
	            if mme2.R.val isa Number
	                if !isfinite(mme2.R.val) || mme2.R.val <= 0
	                    error("NNMM: invalid 2->3 residual variance at iter=$iter: $(mme2.R.val)")
	                end
	            else
	                if !all(isfinite, mme2.R.val)
	                    error("NNMM: non-finite 2->3 residual covariance at iter=$iter")
	                end
	            end
	        end


	        #update σ2_yobs, σ2_weightsNN, and weights_NN
	        mme1.σ2_yobs = mme2.R.val
	        mme1.σ2_weightsNN = middle_weight_variances_23()
	        mme1.weights_NN = middle_weights_23()
	        if mme1.σ2_yobs isa Number
	            if !isfinite(mme1.σ2_yobs) || mme1.σ2_yobs <= 0
	                error("NNMM: invalid σ2_yobs propagated from 2->3 at iter=$iter: $(mme1.σ2_yobs)")
	            end
	        end
	        if mme1.σ2_weightsNN isa Number
	            if !isfinite(mme1.σ2_weightsNN) || mme1.σ2_weightsNN <= 0
	                error("NNMM: invalid σ2_weightsNN propagated from 2->3 at iter=$iter: $(mme1.σ2_weightsNN)")
	            end
	        end
	        if any(x -> !isfinite(x), mme1.weights_NN)
	            error("NNMM: non-finite NN weights propagated from 2->3 at iter=$iter")
	        end

	        ########################################################################
	        # 3.1 Save MCMC samples (model parameters, marker effects, hyperparameters)
	        ########################################################################
	        if iter > burnin && (iter - burnin) % output_samples_frequency == 0
	            #MCMC samples from posterior distributions
	            nsamples       = (iter-burnin)/output_samples_frequency
	            output_posterior_mean_variance(mme1,nsamples)
	            output_posterior_mean_variance(mme2,nsamples)
	            #mean and variance of posterior distribution
	            if mme1.pedTrmVec!=0
	                polygenic_effects_variance = inv(mme1.rndTrmVec[polygenic_pos].Gi.val) 
	            else
	                polygenic_effects_variance=false 
	            end
	            # EBV direct skip (Layer 1 -> Layer 3) contribution on output IDs.
	            direct_skip = false
	            if mme1.nonlinear_function != false && mme1.output_ID != 0 && mme2.M != 0
	                n_out = length(mme1.output_ID)
	                has_skip = any(Mi isa Genotypes for Mi in mme2.M)
	                if has_skip
	                    direct_skip_vec = zeros(n_out)
	                    for Mi in mme2.M
	                        if Mi isa Genotypes
	                            if Mi.output_genotypes == false
	                                error("NNMM: missing output genotypes for 2->3 genotype-skip term; cannot compute EBV_Direct_Skip")
	                            end
	                            if size(Mi.output_genotypes, 1) != n_out
	                                error("NNMM: output genotypes for 2->3 genotype-skip term have $(size(Mi.output_genotypes,1)) rows, expected $n_out")
	                            end
	                            direct_skip_vec .+= Mi.output_genotypes * Mi.α[1]
	                        end
	                    end
	                    direct_skip = direct_skip_vec
	                end
	            end
	            output_MCMC_samples(mme1,mme1.R.val,polygenic_effects_variance,outfile; direct_skip=direct_skip)
	            #  if causal_structure != false
	            #      writedlm(causal_structure_outfile,sample4λ_vec',',')
	            #  end
            # Save Layer 2 variances
            writedlm(outfile["layer2_residual_variance"], mme2.R.val', ',')
            if mme2.M != 0 && length(mme2.M) > 0
                writedlm(outfile["layer2_effect_variance"], middle_weight_variances_23()', ',')
            end
	        end

	        ########################################################################
	        # 3.2 Save prediction samples (EPV)
	        #
	        # Allow EPV/EPV_Output to be saved more frequently than marker effects to
	        # reduce Monte Carlo error in posterior-mean predictions without dumping
	        # huge marker-effect files each iteration.
	        ########################################################################
	        if iter > burnin && (iter - burnin) % output_prediction_frequency == 0
	            # Save EPV (Estimated Phenotypic Value) using OBSERVED omics.
	            # EPV = activation(observed_omics) * weights_NN (+ genotype-skip terms if present)
	            if mme1.nonlinear_function != false && mme2.M != 0 && length(mme2.M) > 0 && haskey(outfile, "EPV_NonLinear")
	                observed_omics = middle_matrix_phenotype_order_23()
	                if mme1.is_activation_fcn == true
	                    # `aligned_omics_w_phenotype` is already activation-transformed (see align_transformed_omics_with_phenotypes),
	                    # so do NOT apply the activation function a second time.
	                    EPV_NN = observed_omics * mme1.weights_NN
	                else
	                    EPV_NN = mme1.nonlinear_function.(Tuple([view(observed_omics, :, i) for i in 1:size(observed_omics, 2)])...)
	                end
	                # Add optional genotype-skip contributions (Layer 1 → Layer 3 direct term).
	                if length(mme2.M) > 1
	                    for Mi in mme2.M
	                        if Mi isa Genotypes
	                            EPV_NN .+= Mi.genotypes * Mi.α[1]
	                        end
	                    end
	                end
	                writedlm(outfile["EPV_NonLinear"], EPV_NN', ',')
	            end

	            # Save EPV on output IDs (includes test individuals even if phenotype is missing).
	            if mme1.output_ID != 0 && haskey(outfile, "EPV_Output_NonLinear")
	                # `ylats_old` is the current latent/observed omics matrix in mme1.obsID order.
	                # Align to mme1.output_ID order if needed, then compute EPV under the same
	                # nonlinearity settings used by EBV_NonLinear.
	                omics_out = ylats_old
	                if Z_output_from_obs !== nothing
	                    omics_out = Z_output_from_obs * omics_out
	                end
	                if mme1.is_activation_fcn == true
	                    omics_out = nonlinear_function.(omics_out)
	                    EPV_out = omics_out * mme1.weights_NN
	                else
	                    EPV_out = mme1.nonlinear_function.(Tuple([view(omics_out, :, i) for i in 1:size(omics_out, 2)])...)
	                end
	                # Add genotype-skip contributions on output IDs.
	                if mme2.M != 0 && length(mme2.M) > 1
	                    n_out = length(mme1.output_ID)
	                    for Mi in mme2.M
	                        if Mi isa Genotypes
	                            if Mi.output_genotypes == false
	                                error("NNMM: missing output genotypes for 2->3 genotype-skip term; cannot compute EPV_Output_NonLinear")
	                            end
	                            if size(Mi.output_genotypes, 1) != n_out
	                                error("NNMM: output genotypes for 2->3 genotype-skip term have $(size(Mi.output_genotypes,1)) rows, expected $n_out")
	                            end
	                            EPV_out .+= Mi.output_genotypes * Mi.α[1]
	                        end
	                    end
	                end
	                writedlm(outfile["EPV_Output_NonLinear"], EPV_out', ',')
	            end
	        end
	        ########################################################################
	        # 3.3 Printout
	        ########################################################################
        if iter%mme1.MCMCinfo.printout_frequency==0 && iter>burnin
            println("\nPosterior means at iteration: ",iter)
            # Print residual variance in a clean format
            if mme1.R.constraint == true && mme1.nModels > 1
                println("Residual variance (diagonal):")
                max_traits = 5
                for t in 1:min(mme1.nModels, max_traits)
                    @printf("  trait %d: %.6f\n", t, mme1.meanVare[t,t])
                end
                if mme1.nModels > max_traits
                    println("  ... (", mme1.nModels - max_traits, " more traits)")
                end
            else
                println("Residual variance: ",round.(mme1.meanVare,digits=6))
            end
        end
    end

    ############################################################################
    # After MCMC
    ############################################################################
    if output_samples_frequency != 0
      for (key,value) in outfile
        close(value)
      end
      if causal_structure != false
        close(causal_structure_outfile)
      end
    end
    if methods == "GBLUP"
        for Mi in mme1.M
            mv(output_folder*"/MCMC_samples_marker_effects_variances"*"_"*Mi.name*".txt",
               output_folder*"/MCMC_samples_genetic_variance(REML)"*"_"*Mi.name*".txt")
        end
    end

    output=output_result(mme1,output_folder,
                         mme1.solMean,mme1.meanVare,
                         mme1.pedTrmVec!=0 ? mme1.G0Mean : false,
                         mme1.solMean2,mme1.meanVare2,
                         mme1.pedTrmVec!=0 ? mme1.G0Mean2 : false)
    output_ID2 = mme2.output_ID
    mme2.output_ID = 0
    output2=output_result(mme2,output_folder,
                          mme2.solMean,mme2.meanVare,
                          mme2.pedTrmVec!=0 ? mme2.G0Mean : false,
                          mme2.solMean2,mme2.meanVare2,
                          mme2.pedTrmVec!=0 ? mme2.G0Mean2 : false)
    mme2.output_ID = output_ID2
    for (key,value) in output2
        if startswith(key, "marker effects ")
            output[key] = value
        end
    end
    return output
end
