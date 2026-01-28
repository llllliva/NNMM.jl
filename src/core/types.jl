#=
================================================================================
Core Type Definitions for Mixed Model Equations
================================================================================
Fundamental data structures for building and solving mixed models.

Types defined here:
- ModelTerm: Individual term in a model equation
- Variance: Variance/covariance parameter container
- ResVar: Residual variance structure (multi-trait)
- RandomEffect: Random effect specification
- Genotypes: Genotype data and marker effect parameters
- MME: Mixed Model Equations container
- MCMCinfo: MCMC sampler configuration

These types are used internally by NNMM.jl for model construction
and MCMC sampling. Most users will interact with these through
the higher-level `Layer` and `Equation` types.

Author: NNMM.jl Team (adapted from JWAS.jl)
================================================================================
=#

"""
    ModelTerm

A struct representing a single term in a model equation.

# Examples
For model equations like "y1 = A + B" and "y2 = A + B + A*B",
ModelTerm instances would be created for terms: y1:A, y1:B, y2:A, y2:B, y2:A*B

# Fields
- `iModel`: Model equation number (1st, 2nd, etc.)
- `iTrait`: Trait name ("y1", "y2", etc.)
- `trmStr`: Term string (e.g., "y1:A")
- `nFactors`: Number of factors in the term
- `factors`: Array of factor symbols
- `data`: String representations of factor levels
- `val`: Numeric values for the term
- `nLevels`: Number of levels for this term
- `names`: Level names
- `startPos`: Start position in incidence matrix
- `X`: Incidence matrix
- `random_type`: "fixed" or "random"
"""
mutable struct ModelTerm
    iModel::Int64                  # 1st (1) or 2nd (2) model_equation
    iTrait::AbstractString         # trait 1 ("y1") or trait 2 ("y2") (trait name)
                                   # | trmStr  | nFactors  | factors |
                                   # |---------|-----------|---------|
    trmStr::AbstractString         # | "y1:A"  |     1     | :A      |
    nFactors::Int64                # | "y2:A"  |     1     | :A      |
    factors::Array{Symbol,1}       # | "y1:A*B"|     2     | :A,:B   |

                                                     #DATA             |          str               |     val       |
                                                     #                :|----------------------------|---------------|
    data::Array{AbstractString,1}                    #covariate^2     :|["A x B", "A X B", ...]     | df[:A].*df[:B]|
    val::Union{Array{Float64,1},Array{Float32,1}}    #factor^2        :|["A1 x B1", "A2 X B2", ...] | [1.0,1.0,...] |
                                                     #factor*covariate:|["A1 x B","A2 X B", ...]    | 1.0.*df[:B]   |

                                   #OUTPUT           | nLevels |     names        |
                                   #                 |---------|------------------|
    nLevels::Int64                 #covariate   :    | 1       | "A"              |
    names::Array{Any,1}            #factor      :    | nLevels | "A1", "A2", ...  |
                                   #animal (ped):    | nAnimals| ids              |
                                   #animal(ped)*age: | nAnimals| "A1*age","A2*age"|
                                   #factor*covariate:| nLevels | "A1*age","A2*age"|

    startPos::Int64                         #start postion for this term in incidence matrix
    X                                       #incidence matrix

    random_type::String

    function ModelTerm(trmStr,m,traitname)
        iModel    = m
        trmStr    = strip(trmStr)
        traitname = strip(traitname)
        factorVec = split(trmStr,"*")
        nFactors  = length(factorVec)
        factors   = [Symbol(strip(f)) for f in factorVec]
        trmStr    = traitname*":"*trmStr
        new(iModel,traitname,trmStr,nFactors,factors,[],zeros(1),0,[],0,false,"fixed")
    end
end

"""
    Variance

A struct for storing variance/covariance parameters used throughout NNMM.

Used for:
- Marker effect variances
- Residual variances
- Variance for non-genetic random effects

# Fields
- `val`: Variance value (Float for single-trait, Matrix for multi-trait, or Bool=false if unset)
- `df`: Degrees of freedom for prior distribution
- `scale`: Scale parameter for prior distribution
- `estimate_variance`: Whether to estimate variance at each MCMC iteration
- `estimate_scale`: Whether to estimate scale at each MCMC iteration
- `constraint`: If true (multi-trait only), covariance between traits is zero

# Example
```julia
v = Variance(1.5, 4.0, 1.3, true, false, false)
```
"""
mutable struct Variance
    val::Union{AbstractFloat, AbstractArray, Bool}   #value of the variance, e.g., single-trait: 1.5, two-trait: [1.3 0.4; 0.4 0.8]
    df::Union{AbstractFloat,Bool}                    #degrees of freedom, e.g., 4.0
    scale::Union{AbstractFloat, AbstractArray, Bool} #scale, e.g., single-trait: 1.0, two-trait: [1.0 0; 0 1.0]

    estimate_variance::Bool  #estimate_variance=true means estimate variance at each MCMC iteration
    estimate_scale::Bool     #estimate_scale=true means estimate scale at each MCMC iteration
    constraint::Bool        #constraint=true means in multi-trait analysis, covariance is zero
end



"""
    ResVar

A struct for residual covariance matrix for all observations of size (nobs × nModels).

The Ri matrix is modified based on missing pattern (number of Ri = 2^ntraits - 1).
This allows using the same incidence matrix X for all traits in multi-trait analyses.

In NNMM, this is ONLY used when:
- Residual variance is constant, or
- Missing phenotypes are not imputed at each MCMC step (no marker effects)

# Fields
- `R0`: Base residual covariance matrix
- `RiDict`: Dictionary mapping missing patterns to adjusted covariance matrices
"""
mutable struct ResVar
    R0::Union{Array{Float64,2},Array{Float32,2}}
    RiDict::Dict{BitArray{1},Union{Array{Float64,2},Array{Float32,2}}}
end

"""
    RandomEffect

A struct for general (including i.i.d.) random effects.

Assumes independence: cov(1:A, 1:B) = 0 unless A.names == B.names (e.g., pedigree terms).

# Examples
- Single-trait: termarray = [ModelTerm(1:A)]
- Multi-trait: termarray = [ModelTerm(1:A), ModelTerm(2:A)]

# Fields
- `term_array`: Array of term strings
- `Gi`: Covariance matrix (Variance object)
- `GiOld`: Previous iteration value (for lambda version of MME)
- `GiNew`: Current iteration value (for lambda version of MME)
- `Vinv`: Inverse of relationship/covariance structure (0 for identity)
- `names`: Level names/IDs
- `randomType`: Type of random effect ("A" for additive genetic, etc.)
"""
mutable struct RandomEffect
    term_array::Array{AbstractString,1}
    Gi     #covariance matrix (multi-trait) #the "Variance" object
    GiOld  #specific for lambda version of MME (single-trait) #Variance.val has type ::Array{Float64,2}
    GiNew  #specific for lambda version of MME (single-trait) #Variance.val has type ::Array{Float64,2}
    # df::AbstractFloat
    # scale #::Array{Float64,2}
    Vinv # 0, identity matrix
    names #[] General IDs and Vinv matrix (order is important now)(modelterm.names)
    randomType::String
end

"""
    Genotypes

A struct for storing genotype data and associated parameters for genomic prediction.

# Fields
- `name`: Category name (e.g., "geno1")
- `trait_names`: Names of corresponding traits (e.g., ["y1", "y2"])
- `obsID`: Individual IDs for genotyped and phenotyped individuals
- `markerID`: Marker/SNP identifiers
- `nObs`: Number of observations
- `nMarkers`: Number of markers
- `alleleFreq`: Allele frequencies
- `sum2pq`: Sum of 2*p*q for all markers
- `centered`: Whether genotypes are centered
- `genotypes`: Genotype matrix
- `genetic_variance`: Genetic variance (Variance struct)
- `G`: Marker effect variance (Variance struct)
- `method`: Bayesian method ("BayesA", "BayesB", "BayesC", etc.)
- `estimatePi`: Whether to estimate Pi
- MCMC-related arrays (α, β, δ, π, etc.)
- Result arrays (meanAlpha, meanVara, etc.)
"""
mutable struct Genotypes
  name                            # Name for this category, e.g., "geno1"
  trait_names                     # Names for corresponding traits, e.g., ["y1","y2"]

  obsID::Array{AbstractString,1}  #row ID for (imputed) genotyped and phenotyped inds (finally)
  markerID
  nObs::Int64                     #length of obsID
  nMarkers::Int64
  alleleFreq
  sum2pq::AbstractFloat
  centered::Bool
  genotypes::Union{Array{Float64,2},Array{Float32,2}}
  nLoci             #number of markers included in the model
  ntraits           #number of traits included in the model

  genetic_variance  #genetic variance, type: Variance struct
  G       #marker effect variance; the "Variance" object, for Variance.val: ST->Float64;MT->Array{Float64,2}
#   scale             #scale parameter for marker effect variance (G)
#   df                #degree of freedom

  method            #prior for marker effects (Bayesian ALphabet, GBLUP ...)
  estimatePi
#   estimate_variance
#   estimate_scale

  mArray            #a collection of matrices used in Bayesian Alphabet
  mRinvArray        #a collection of matrices used in Bayesian Alphabet
  mpRinvm           #a collection of matrices used in Bayesian Alphabet
  mΦΦArray          # a collection of matrices used in RRM
  MArray            #a collection of matrices used in Bayesian Alphabet (rhs approach)
  MRinvArray        #a collection of matrices used in Bayesian Alphabet (rhs approach)
  MpRinvM           #a collection of matrices used in Bayesian Alphabet (rhs approach)
  D                 #eigen values used in GBLUP
  gammaArray        #array used in Bayesian LASSO

  α                 #array of current MCMC samples
  β
  δ
  π

  meanAlpha         #arrays of results
  meanAlpha2
  meanDelta
  mean_pi
  mean_pi2
  meanVara
  meanVara2
  meanScaleVara
  meanScaleVara2

  output_genotypes #output genotypes

  isGRM  #whether genotypes or relationship matirx is provided

  Genotypes(a1,a2,a3,a4,a5,a6,a7,a8,a9)=new(false,false,
                                         a1,a2,a3,a4,a5,a6,a7,a8,a4,false,
                                         Variance(false,false,false,true,false,false),Variance(false,false,false,true,false,false), #false,false,
                                         false,true, #true,false,
                                         false,false,false,false,false,false,false,false,false,
                                         false,false,false,false,
                                         false,false,false,false,false,false,false,false,false,
                                         false,a9)
end

"""
    MCMCinfo

A struct for storing MCMC run configuration and parameters.

# Fields
- `heterogeneous_residuals`: Whether to use heterogeneous residual variances
- `chain_length`: Total number of MCMC iterations
- `burnin`: Number of burn-in iterations to discard
- `output_samples_frequency`: How often to save MCMC samples
- `output_prediction_frequency`: How often to save EBV/EPV prediction samples (may be more frequent than marker-effect samples)
- `printout_model_info`: Whether to print model information
- `printout_frequency`: How often to print progress
- `single_step_analysis`: Whether using single-step genomic evaluation
- `fitting_J_vector`: Whether to fit J vector in single-step
- `missing_phenotypes`: Whether to handle missing phenotypes
- `update_priors_frequency`: How often to update priors
- `outputEBV`: Whether to output estimated breeding values
- `output_heritability`: Whether to output heritability estimates
- `prediction_equation`: Custom prediction equation
- `seed`: Random seed for reproducibility
- `double_precision`: Whether to use Float64 precision
- `output_folder`: Directory for output files
- `RRM`: Random regression model parameters
- `fast_blocks`: Block size for fast sampling
"""
mutable struct MCMCinfo
    heterogeneous_residuals
    chain_length
    burnin
    output_samples_frequency
    output_prediction_frequency
    printout_model_info
    printout_frequency
    single_step_analysis
    fitting_J_vector
    missing_phenotypes
    # constraint
    # mega_trait
    # estimate_variance
    update_priors_frequency
    outputEBV
    output_heritability
    prediction_equation
    seed
    double_precision
    output_folder
    RRM
    fast_blocks
end
"""
    MME

Mixed Model Equations - the main struct for Bayesian Linear Mixed Models.

Supports:
- Single-trait analysis: lambda version of MME
- Multi-trait analysis: formal version of MME

Scale parameters for variance components are computed when variances are added:
- `build_model()` for residual variance
- `set_random()` for non-marker random effects
- `set_marker_hyperparameters_variances_and_pi()` for marker effect variance

# Key Fields
- `nModels`: Number of model equations (traits)
- `modelVec`: Array of model equation strings
- `modelTerms`: Array of ModelTerm objects
- `modelTermDict`: Dictionary mapping term strings to ModelTerm objects
- `lhsVec`: Phenotype symbols (e.g., [:y1, :y2])
- `covVec`: Covariate symbols
- `X`: Incidence matrix
- `ySparse`: Phenotype vector
- `obsID`: Individual IDs
- `mmeLhs`, `mmeRhs`: Left and right hand sides of MME
- `pedTrmVec`: Polygenic effect terms from pedigree
- `ped`: Pedigree object
- `rndTrmVec`: General random effects
- `R`: Residual variance (Variance struct)
- `M`: Genotypes object(s)
- `MCMCinfo`: MCMC configuration
- Neural network fields for NNMM functionality
"""
mutable struct MME
    nModels::Integer                              #number of model equations
    modelVec::Array{AbstractString,1}             #["y1 = A + B","y2 = A + B + A*B"]
    modelTerms::Array{ModelTerm,1}                #ModelTerms for "1:intercept","1:A","2:intercept","2:A","2:A*B"...;
    modelTermDict::Dict{AbstractString,ModelTerm} #key: "1:A*B" value: ModelTerm; convert modelTerms above to dictionary
    lhsVec::Array{Symbol,1}                       #phenotypes: [:y1,:y2]
    covVec::Array{Symbol,1}                       #variables those are covariates

                                                  #MIXED MODEL EQUATIONS
    X                                             #incidence matrix
    ySparse                                       #phenotypes
    obsID::Array{AbstractString,1}                #IDs for phenotypes
    mmeLhs                                        #Lhs of Mixed Model Equations
    mmeRhs                                        #Rhs of Mixed Model Equations

                                                  #RANDOM EFFCTS
    pedTrmVec                                     #polygenic effects(pedigree): "1:Animal","1:Mat","2:Animal"
    ped                                           #PedModule.Pedigree
    Gi                                            #inverse of genetic covariance matrix for pedTrmVec (multi-trait)
    GiOld                                         #specific for lambda version of MME (single-trait)
    GiNew                                         #specific for lambda version of MME (single-trait)
    scalePed
    G0Mean
    G0Mean2

    rndTrmVec::Array{RandomEffect,1}              #General (including i.i.d.) random effects
                                                  #may merge pedTrmVec here

                                                  #RESIDUAL EFFECTS
    R::Variance                                   #residual covariance matrix (multi-trait) ::Array{Union{Float64,Float32},2}
    missingPattern                                #for impuation of missing residual
    resVar                                        #for impuation of missing residual
    ROld                #initilized to 0 ??       #residual variance (single-trait) for
    # scaleR                                      #scale parameters
    meanVare
    meanVare2

    invweights                                    #heterogeneous residuals

    M                                             #GENOTYPES

    mmePos::Integer                               #temporary value to record term position (start from 1)

    outputSamplesVec::Array{ModelTerm,1}          #for which location parameters to save MCMC samples

    # df::DF                                        #prior degree of freedom

    output_ID
    output_genotypes
    output_X

    output

    MCMCinfo

    sol
    solMean
    solMean2

    causal_structure

    nonlinear_function #user-provide function, "tanh"
    weights_NN
    σ2_yobs
    is_fully_connected
    is_activation_fcn  #Neural Network with activation function (not user-defined function)
    latent_traits #["z1","z2"], for intermediate omics data,
    yobs          #for single observed trait, and mme.ySparse is for latent traits
    yobs_name
    σ2_weightsNN
    fixed_σ2_NN
    incomplete_omics

    traits_type   #by default all traits are continuous
    thresholds    #thresholds for categorial&binary traits. Dictionary: 1=>[-Inf,0,Inf], where 1 means the 1st trait

    function MME(nModels,modelVec,modelTerms,dict,lhsVec,R) #MME(nModels,modelVec,modelTerms,dict,lhsVec,R,ν)
        return new(nModels,modelVec,modelTerms,dict,lhsVec,[],
                   0,0,[],0,0,
                   0,0,zeros(1,1),zeros(1,1),zeros(1,1),zeros(1,1),false,false,
                   [],
                   R,0,0,R.val,false,false, #starting value of mme.ROld is a scalar equal to mme.R.val
                   [],
                   0,
                   1,
                   [],
                #    DF(νR0,4,4,4),
                   0,0,Dict{String,Any}(),
                   0,
                   0,
                   false,false,false,
                   false,
                   false,false,1.0,false,false,false,false,false,1.0/sqrt(nModels),false,false,
                   repeat(["continuous"],nModels),Dict())
    end
end
