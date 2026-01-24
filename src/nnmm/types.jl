#=
================================================================================
NNMM Type Definitions
================================================================================
Core data structures for the Neural Network Mixed Model.

Types defined here:
- Layer: Network layer specification (genotypes, omics, phenotypes)
- Equation: Model equation specification
- Omics: Omics/intermediate trait data container
- Phenotypes: Phenotype data container

Author: NNMM.jl Team
================================================================================
=#

"""
    Layer

A struct representing a layer in the Neural Network Mixed Model (NNMM).

# Fields
- `layer_name::String`: Name identifier for the layer
- `data_path`: Path to the data file(s) - String for single file, Vector{String} for partial connection
- `separator`: Column separator in the data file (default: ',')
- `header`: Whether the data file has a header row (default: true)
- `data`: Loaded data (initialized as empty)
- `quality_control`: Whether to perform quality control on genotypes (default: true)
- `MAF`: Minor Allele Frequency threshold for QC (default: 0.01)
- `missing_value`: Value representing missing data (default: 9.0)
- `center`: Whether to center the data (default: true)
"""
mutable struct Layer
    layer_name::String  
    data_path::Union{String, Vector{String}}
    separator::Char
    header::Bool
    data  # Can be Vector{Any}, DataFrame, or other types
    quality_control::Bool
    MAF::Float64
    missing_value::Union{Float64, String}
    center::Bool
    
    function Layer(;layer_name::String, 
                    data_path::Union{String, Vector{String}},
                    separator::Char=',',
                    header::Bool=true, 
                    data=Any[],
                    quality_control::Bool=true,
                    MAF::Float64=0.01,
                    missing_value::Union{Float64, String}=9.0,
                    center::Bool=true)
        new(layer_name, data_path, separator, header, data, 
            quality_control, MAF, missing_value, center)
    end
end

"""
    Omics

A struct representing omics data (e.g., transcriptomics, metabolomics) in NNMM.

# Fields
- `name`: Name identifier for this omics category
- `trait_names`: Names for corresponding traits
- `obsID`: Row IDs for individuals
- `featureID`: Feature/omics variable IDs
- `nObs`: Number of observations
- `nFeatures`: Number of omics features
- `nMarkers`: Alias for nFeatures (compatibility)
- `centered`: Whether data is centered
- `data`: The omics data matrix
- `ntraits`: Number of traits in the model
- Additional fields for Bayesian MCMC computations
"""
mutable struct Omics
    name                            # Name for this category, e.g., "omics1"
    trait_names                     # Names for corresponding traits, e.g., ["y1","y2"]
  
    obsID::Vector{AbstractString}   # Row ID for individuals
    featureID                       # Feature/omics IDs
    nObs::Int64                     # Length of obsID
    nFeatures::Int64                # Number of omics features
    nMarkers::Int64                 # Alias for nFeatures (compatibility with Genotypes)
    centered::Bool
    data                            # Omics data matrix
    ntraits                         # Number of traits in the model
  
    genetic_variance                # Genetic variance (Variance struct)
    G                               # Marker effect variance (Variance object)
  
    method                          # Prior for effects (Bayesian Alphabet, GBLUP, etc.)
    estimatePi::Bool
  
    # Matrices used in Bayesian Alphabet
    mArray
    mRinvArray
    mpRinvm
    mΦΦArray                        # Used in RRM
    D                               # Eigen values for GBLUP
    gammaArray                      # Array for Bayesian LASSO
  
    # Current MCMC samples
    α
    β
    δ
    π
  
    # Result arrays (posterior means)
    meanAlpha
    meanAlpha2
    meanDelta
    mean_pi
    mean_pi2
    meanVara
    meanVara2
    meanScaleVara
    meanScaleVara2
  
    output_genotypes
    isGRM::Bool

    # Alignment with phenotype data
    aligned_omics_w_phenotype
    aligned_obsID_w_phenotype
    aligned_nObs_w_phenotype
    
    function Omics(obsID::Vector{<:AbstractString}, featureID, nObs::Int64, nFeatures::Int64, data)
        new(false, false,
            obsID, featureID, nObs, nFeatures, nFeatures, false, data, false,
            Variance(false, false, false, true, false, false),
            Variance(false, false, false, true, false, false), 
            false, true,
            false, false, false, false, false, false,
            false, false, false, false,
            false, false, false, false, false, false, false, false, false,
            false, false,
            false, false, false)
    end
end

"""
    Phenotypes

A struct representing phenotype data in NNMM.

# Fields
- `obsID`: Individual IDs
- `featureID`: Phenotype/trait names
- `nObs`: Number of observations
- `nPheno`: Number of phenotypes
- `data`: Phenotype data matrix
- `nFeatures`: Alias for nPheno
"""
mutable struct Phenotypes
    obsID::Vector{AbstractString}       # Individual IDs
    featureID::Vector{AbstractString}   # Phenotype names
    nObs::Int64                         # Number of observations
    nPheno::Int64                       # Number of phenotypes
    data                                # Phenotype data matrix
    nFeatures::Int64                    # = nPheno
    
    function Phenotypes(obsID::Vector{<:AbstractString}, featureID::Vector{<:AbstractString}, 
                       nObs::Int64, nPheno::Int64, data)
        new(obsID, featureID, nObs, nPheno, data, nPheno)
    end
end

"""
    Equation

A struct representing an equation connecting layers in NNMM.

# Fields
- `from_layer_name`: Source layer name
- `to_layer_name`: Target layer name
- `equation`: Model equation string (e.g., "omics = intercept + geno")
- `traits`: Output trait column names (e.g., ["omic1", "omic2"] or ["trait1"])
- `covariate`: Covariate variable names
- `random`: Random effect specifications
- `activation_function`: Activation function - string ("linear", "sigmoid", "tanh", "relu", "leakyrelu") or custom Function
- `partial_connect_structure`: Structure for partial connectivity
- `starting_value`: Starting values for MCMC
- `method`: Bayesian method ("BayesC", "BayesA", "BayesB", etc.)
- `Pi`: Prior probability of zero effect
- `estimatePi`: Whether to estimate Pi
- Variance parameters for G (genetic) and R (residual)

Note: `omics_name` and `phenotype_name` are deprecated aliases for `traits`.
"""
mutable struct Equation
    from_layer_name::String 
    to_layer_name::String
    equation::String
    traits                    # Output trait column names (replaces omics_name/phenotype_name)
    covariate
    random
    activation_function::Union{String, Function}
    partial_connect_structure
    starting_value
    # Method parameters
    method::String
    Pi::Float64
    estimatePi::Bool
    # Genetic variance parameters
    G
    G_is_marker_variance::Bool
    df_G::Float64
    estimate_variance_G::Bool
    estimate_scale_G::Bool
    constraint_G::Bool
    # Residual variance parameters
    R
    df_R::Float64
    estimate_variance_R::Bool
    estimate_scale_R::Bool
    constraint_R::Bool
    # Optional per-class priors for multi-class marker models (e.g., MiddleLayer + Genotypes in 2->3)
    class_priors
    
    function Equation(;from_layer_name::String, 
                      to_layer_name::String, 
                      equation::String,
                      traits=false,           # New unified parameter
                      omics_name=false,       # Deprecated alias for traits
                      phenotype_name=false,   # Deprecated alias for traits
                      covariate=false,
                      random=false,
                      activation_function::Union{String, Function}="linear",
                      partial_connect_structure=false,
                      starting_value=false,
                      method::String="BayesC", 
                      Pi::Float64=0.0,
                      estimatePi::Bool=true,
                      G=false,
                      G_is_marker_variance::Bool=false,
                      df_G::Float64=4.0,
                      estimate_variance_G::Bool=true,
                      estimate_scale_G::Bool=false,
                      constraint_G::Bool=true,
                      R=false,
                      df_R::Float64=4.0,
                      estimate_variance_R::Bool=true,
                      estimate_scale_R::Bool=false,
                      constraint_R::Bool=true,
                      class_priors=false)
        
        # Handle deprecated aliases (omics_name, phenotype_name) for backward compatibility
        if traits == false
            if omics_name != false
                traits = omics_name
            elseif phenotype_name != false
                traits = phenotype_name
            else
                error("traits must be provided (specifies output trait column names).")
            end
        end
        
        new(from_layer_name, to_layer_name, equation,       
            traits, covariate, random,
            activation_function, partial_connect_structure, starting_value,
            method, Pi, estimatePi,
            G, G_is_marker_variance, df_G, estimate_variance_G, estimate_scale_G, constraint_G,
            R, df_R, estimate_variance_R, estimate_scale_R, constraint_R,
            class_priors)    
    end
end
