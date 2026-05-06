using Test
using NNMM

@testset "NNMM.jl" begin
    # Core type tests
    include("test_types.jl")
    
    # Utility function tests
    include("test_utilities.jl")
    
    # Pedigree module tests
    include("test_pedigree.jl")
    
    # Genotype loading tests
    include("test_genotypes.jl")
    
    # NNMM basic workflow tests
    include("test_nnmm_basic.jl")
    
    # NNMM full workflow with pedigree and covariates
    include("test_nnmm_full.jl")
    
    # NNMM activation function tests
    include("test_nnmm_activation.jl")
    
    # NNMM missing data handling tests
    include("test_nnmm_missing.jl")
    
    # NNMM Bayesian method tests
    include("test_nnmm_methods.jl")
    
    # NNMM as traditional BayesC (missing middle layer pattern)
    include("test_nnmm_as_bayesc.jl")
    
    # Additional Bayesian methods (BayesB, BayesL, RR-BLUP)
    include("test_more_methods.jl")
    
    # Post-analysis functions (GWAS, getEBV)
    include("test_post_analysis.jl")

    # Internal debug invariants / regression checks
    include("test_invariants.jl")

    # Layer 1 → Layer 3 skip connection (multi-class 2->3)
    include("test_skip_connection.jl")

    # Named omics groups as separate 2->3 marker classes
    include("test_omics_groups.jl")
end
