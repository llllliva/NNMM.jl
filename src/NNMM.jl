__precompile__(true)

"""
    NNMM

Neural Network Mixed Model for genomic prediction with intermediate omics data.

# Overview
NNMM implements a two-layer neural network framework for genomic prediction:

    Layer 1 (Genotypes) → Layer 2 (Omics/Latent) → Layer 3 (Phenotypes)

The model uses Bayesian methods (BayesA, BayesB, BayesC, RR-BLUP, BayesL, GBLUP)
for marker effect estimation and HMC/Metropolis-Hastings for sampling latent traits.

# Quick Start
```julia
using NNMM

# Define network layers
layer1 = Layer(layer_name="geno",  data_path="genotypes.csv")
layer2 = Layer(layer_name="omics", data_path="omics.csv")
layer3 = Layer(layer_name="pheno", data_path="phenotypes.csv")

# Define equations
eq1 = Equation("omics = intercept + geno", 
               method="BayesC", omics_name=["o1","o2"])
eq2 = Equation("pheno = intercept + omics",
               activation_function=tanh, phenotype_name=["y"])

# Run MCMC
results = runNNMM([layer1, layer2, layer3], [eq1, eq2],
                  chain_length=10000, burnin=2000)
```

# Main Types
- `Layer`: Network layer specification
- `Equation`: Model equation with Bayesian method selection

# Main Functions
- `runNNMM`: Run NNMM analysis
- `describe`: Print model information
- `GWAS`: Genome-wide association study on results
- `getEBV`: Extract estimated breeding values

# References
- NNMM methodology paper (forthcoming)
- JWAS.jl: Julia package for whole-genome analyses

Author: NNMM.jl Team
License: MIT
"""
module NNMM

using Distributions, Printf, Random
using DelimitedFiles
using InteractiveUtils  # for versioninfo
using DataFrames, CSV
using SparseArrays
using LinearAlgebra
using ProgressMeter
using ForwardDiff

import StatsBase: describe  # a new describe is exported

"""
    print_matrix_truncated(mat, max_size=5)

* (internal function) Print a matrix, truncating if larger than max_size.
"""
function print_matrix_truncated(mat; max_size=5, digits=3)
    n = size(mat, 1)
    if n <= max_size
        Base.print_matrix(stdout, round.(mat, digits=digits))
    else
        # Show first max_size rows and columns
        Base.print_matrix(stdout, round.(mat[1:max_size, 1:max_size], digits=digits))
        println("\n  ... (showing $(max_size)×$(max_size) of $(n)×$(n) matrix)")
    end
    println()
end

# Pedigree Module (loaded first as other modules depend on it)
include("pedigree/PedModule.jl")
using .PedModule

# Core types and model building
include("core/types.jl")
include("core/build_MME.jl")
include("core/random_effects.jl")
include("core/residual.jl")
include("core/variance_components.jl")

# Iterative Solver (Gibbs sampler used by NNMM MCMC)
include("solvers/iterative_solver.jl")

# NNMM (Neural Network Mixed Models)
include("nnmm/types.jl")
include("nnmm/read_files.jl")
include("nnmm/hmc.jl")
include("nnmm/check.jl")
include("nnmm/run_mcmc.jl")
include("nnmm/mcmc_bayesian.jl")

# Genomic Markers (Bayesian Alphabet methods)
include("markers/genotype_tools.jl")
include("markers/BayesABC.jl")
include("markers/BayesC0L.jl")
include("markers/GBLUP.jl")
include("markers/MTBayesABC.jl")
include("markers/MTBayesC0L.jl")
include("markers/Pi.jl")

# Input/Output
include("io/input_validation.jl")
include("io/output.jl")

# GWAS
include("gwas/GWAS.jl")

# Datasets
include("datasets/Datasets.jl")
using .Datasets: dataset

# =============================================================================
# Exports - Public API (user-facing)
# =============================================================================
# Main NNMM functions
export Layer, Equation, runNNMM, describe

# Data reading functions
export read_phenotypes                      # Read phenotype data from file
export nnmm_get_genotypes, get_genotypes    # Read genotype data (get_genotypes is alias)
export nnmm_get_omics                       # Read omics data

# Built-in datasets
export dataset

# Post-analysis
export GWAS, getEBV

# =============================================================================
# Exports - Advanced/Internal (may be needed by power users)
# =============================================================================
export Omics, Phenotypes                    # Data types (usually not needed directly)
export set_covariate, set_random            # Model building helpers
export outputMCMCsamples, outputEBV         # Output helpers
export get_pedigree, get_info               # Pedigree utilities
export mkmat_incidence_factor               # Internal utility


# =============================================================================
# Model Description Functions
# =============================================================================

"""
    describe(model::MME)

Print a summary of the mixed model equations (MME) structure.

Displays:
- Model equations (truncated if >5)
- Term information (classification, fixed/random, number of levels)
- Prior distributions and hyperparameters
- Variance component settings
- Marker effect settings (if genomic data included)

# Arguments
- `model::MME`: The mixed model equations object (created internally by runNNMM)

# Output
Prints formatted model summary to stdout including:
- Model equations
- Term classification (covariate/factor, fixed/random)
- Prior settings for variance components
- MCMC configuration

# Example
```julia
# After running runNNMM, the describe function is called automatically.
# For manual inspection:
describe(results["mme"])
```
"""
function describe(model::MME)
    printstyled("\nA Linear Mixed Model was build using model equations:\n\n",bold=true)
    max_eqs = 5
    for (idx, eq) in enumerate(model.modelVec)
        if idx > max_eqs
            println("... (", length(model.modelVec) - max_eqs, " more equations)")
            break
        end
        println(eq)
    end
    println()
    printstyled("Model Information:\n\n",bold=true)
    @printf("%-15s %-12s %-10s %11s\n","Term","C/F","F/R","nLevels")

    random_effects=Array{AbstractString,1}()
    if model.pedTrmVec != 0
    for i in model.pedTrmVec
        push!(random_effects,split(i,':')[end])
    end
    end
    for i in model.rndTrmVec
      for j in i.term_array
          push!(random_effects,split(j,':')[end])
      end
    end

    terms=[]
    for i in model.modelTerms
    term    = split(i.trmStr,':')[end]
    if term in terms
        continue
    else
        push!(terms,term)
    end

    nLevels = i.nLevels
    fixed   = (term in random_effects) ? "random" : "fixed"
    factor  = (nLevels==1) ? "covariate" : "factor"

    if term =="intercept"
        factor="factor"
    elseif length(split(term,'*'))!=1
        factor="interaction"
    end

    @printf("%-15s %-12s %-10s %11s\n",term,factor,fixed,nLevels)
    end
    println()
    if model.MCMCinfo != false && model.MCMCinfo.printout_model_info == true
        getMCMCinfo(model)
    end
end

"""
    getMCMCinfo(model::MME)

* (internal function) Print out MCMC information.
"""
function getMCMCinfo(mme)
    is_nnbayes_partial = mme.nonlinear_function != false && mme.is_fully_connected==false
    if mme.MCMCinfo == false
        printstyled("MCMC information is not available\n\n",bold=true)
        return
    end
    MCMCinfo = mme.MCMCinfo
    printstyled("MCMC Information:\n\n",bold=true)
    @printf("%-30s %20s\n","chain_length",MCMCinfo.chain_length)
    @printf("%-30s %20s\n","burnin",MCMCinfo.burnin)
    @printf("%-30s %20s\n","starting_value",mme.sol != false ? "true" : "false")
    @printf("%-30s %20d\n","printout_frequency",MCMCinfo.printout_frequency)
    @printf("%-30s %20d\n","output_samples_frequency",MCMCinfo.output_samples_frequency)
    @printf("%-30s %20d\n","output_prediction_frequency",MCMCinfo.output_prediction_frequency)
    @printf("%-30s %19s\n","constraint on residual variance",mme.R.constraint ? "true" : "false")
    if mme.M != 0
        for Mi in mme.M
            geno_name = Mi.name
            @printf("%-30s %5s\n","constraint on marker effect variance for $geno_name",Mi.G.constraint ? "true" : "false")
        end
    end
    @printf("%-30s %20s\n","missing_phenotypes",MCMCinfo.missing_phenotypes ? "true" : "false")
    @printf("%-30s %20d\n","update_priors_frequency",MCMCinfo.update_priors_frequency)
    @printf("%-30s %20s\n","seed",MCMCinfo.seed)

    printstyled("\nHyper-parameters Information:\n\n",bold=true)
    for i in mme.rndTrmVec
        thisterm= join(i.term_array, ",")
        if mme.nModels == 1
            @printf("%-30s %20s\n","random effect variances ("*thisterm*"):",Float64.(round.(inv(i.GiNew.val),digits=3)))
        elseif i.Gi.constraint == true && mme.nModels > 1
            # FIX: For constraint=true, print only diagonal values
            @printf("%-30s\n","random effect variances ("*thisterm*", diagonal):")
            var_matrix = inv(i.GiNew.val)
            for t in 1:min(mme.nModels, 5)
                @printf("  trait %d: %.4f\n", t, var_matrix[t,t])
            end
            if mme.nModels > 5
                println("  ... (", mme.nModels - 5, " more traits)")
            end
        else
            @printf("%-30s\n","random effect variances ("*thisterm*"):")
            print_matrix_truncated(inv(i.GiNew.val))
        end
    end
    if mme.pedTrmVec!=0
        polygenic_pos = findfirst(i -> i.randomType=="A", mme.rndTrmVec)
    end
    if mme.pedTrmVec!=0
        @printf("%-30s\n","genetic variances (polygenic):")
        print_matrix_truncated(inv(mme.rndTrmVec[polygenic_pos].Gi.val))
    end
    if mme.nModels == 1
        @printf("%-30s %20.3f\n","residual variances:", mme.R.val)
    elseif mme.R.constraint == true && mme.nModels > 1
        # FIX: For constraint=true, print only diagonal values
        @printf("%-30s\n","residual variances (diagonal):")
        for t in 1:min(mme.nModels, 5)
            @printf("  trait %d: %.4f\n", t, mme.R.val[t,t])
        end
        if mme.nModels > 5
            println("  ... (", mme.nModels - 5, " more traits)")
        end
    else
        @printf("%-30s\n","residual variances:")
        print_matrix_truncated(mme.R.val)
    end

    printstyled("\nGenomic Information:\n\n",bold=true)
    if mme.M != false
        print(MCMCinfo.single_step_analysis ? "incomplete genomic data" : "complete genomic data")
        println(MCMCinfo.single_step_analysis ? " (i.e., single-step analysis)" : " (i.e., non-single-step analysis)")
        for Mi in mme.M
            println()
            @printf("%-30s %20s\n","Genomic Category", Mi.name)
            @printf("%-30s %20s\n","Method",Mi.method)
            # FIX: Removed nested loop (was: for Mi in mme.M) that caused N² repeated output
                if Mi.genetic_variance.val != false
                    if (mme.nModels == 1 || is_nnbayes_partial) && mme.MCMCinfo.RRM == false
                        @printf("%-30s %20.3f\n","genetic variances (genomic):",Mi.genetic_variance.val)
                elseif Mi.G.constraint == true && mme.nModels > 1
                    # FIX: For constraint=true (independent traits), print only diagonal values
                    @printf("%-30s\n","genetic variances (genomic, diagonal):")
                    for t in 1:min(mme.nModels, 5)  # Show first 5 traits max
                        @printf("  trait %d: %.4f\n", t, Mi.genetic_variance.val[t,t])
                    end
                    if mme.nModels > 5
                        println("  ... (", mme.nModels - 5, " more traits)")
                    end
                    else
                        @printf("%-30s\n","genetic variances (genomic):")
                    print_matrix_truncated(Mi.genetic_variance.val)
                    end
                end
                if !(Mi.method in ["GBLUP"])
                    if (mme.nModels == 1 || is_nnbayes_partial) && mme.MCMCinfo.RRM == false
                        @printf("%-30s %20.3f\n","marker effect variances:",Mi.G.val)
                elseif Mi.G.constraint == true && mme.nModels > 1
                    # FIX: For constraint=true (independent traits), print only diagonal values
                    @printf("%-30s\n","marker effect variances (diagonal):")
                    for t in 1:min(mme.nModels, 5)  # Show first 5 traits max
                        @printf("  trait %d: %.4f\n", t, Mi.G.val[t,t])
                    end
                    if mme.nModels > 5
                        println("  ... (", mme.nModels - 5, " more traits)")
                    end
                    else
                        @printf("%-30s\n","marker effect variances:")
                    print_matrix_truncated(Mi.G.val)
                    end
                end
                if !(Mi.method in ["RR-BLUP","BayesL","GBLUP"])
                    if mme.nModels == 1 && mme.MCMCinfo.RRM == false
                        @printf("%-30s %20s\n","π",Mi.π)
                elseif Mi.G.constraint == true && mme.nModels > 1
                    # FIX: For constraint=true, π is sampled independently per trait during MCMC
                    # At this point (before MCMC), π may still be in dictionary format
                    # Just print a simple message instead of the full dictionary
                    if Mi.π isa AbstractVector
                        @printf("%-30s\n","π (per trait, independent):")
                        for t in 1:min(mme.nModels, 5)  # Show first 5 traits max
                            @printf("  trait %d: %.4f\n", t, Mi.π[t])
                        end
                        if mme.nModels > 5
                            println("  ... (", mme.nModels - 5, " more traits)")
                        end
                    else
                        # constraint=true but π not yet converted to vector (before MCMC)
                        # Extract initial π value from the dictionary (all values are the same)
                        initial_pi = first(values(Mi.π))
                        @printf("%-30s\n","π (per trait, independent):")
                        if Mi.estimatePi
                            @printf("  prior value: %.4f (will be sampled per-trait)\n", initial_pi)
                        else
                            @printf("  fixed value: %.4f (not sampled)\n", initial_pi)
                        end
                    end
                else
                    # Full multi-trait with covariance: π is a dictionary with 2^ntraits combinations
                        println("\nΠ: (Y(yes):included; N(no):excluded)\n")
                        print(string.(mme.lhsVec))
                        @printf("%20s\n","probability")
                    pi_entries = collect(Mi.π)
                    max_pi_entries = 20  # Truncate if more than 20 entries (2^4 = 16 traits)
                    for idx in 1:min(length(pi_entries), max_pi_entries)
                        (i,j) = pi_entries[idx]
                            i = replace(string.(i),"1.0"=>"Y","0.0"=>"N")
                            print(i)
                            @printf("%20s\n",j)
                        end
                    if length(pi_entries) > max_pi_entries
                        println("  ... (", length(pi_entries) - max_pi_entries, " more combinations)")
                    end
                    println()
                end
                @printf("%-30s %20s\n","estimatePi",Mi.estimatePi ? "true" : "false")
            end
            @printf("%-30s %20s\n","estimate_scale",Mi.G.estimate_scale ? "true" : "false")
        end
    end
    printstyled("\nDegree of freedom for hyper-parameters:\n\n",bold=true)
    @printf("%-30s %20.3f\n","residual variances:",mme.R.df)
    for randomeffect in mme.rndTrmVec
        if randomeffect.randomType != "A"
            @printf("%-30s %20.3f\n","random effect variances:",randomeffect.Gi.df)
        end
    end
    if mme.pedTrmVec!=0
        @printf("%-30s %20.3f\n","polygenic effect variances:",mme.rndTrmVec[polygenic_pos].Gi.df)
    end
    if mme.M!=0
        for Mi in mme.M
            @printf("%-30s %20.3f\n","marker effect variances:",Mi.G.df)
        end
    end
    @printf("\n\n\n")
end

end # module NNMM
