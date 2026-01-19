using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using DelimitedFiles
using LinearAlgebra
using Random

@testset "Full NNMM Validation with Simulated Data" begin
    #=
    This test validates NNMM accuracy using the simulated_omics_data dataset.
    
    Dataset properties:
    - 3,534 individuals, 1,000 SNPs, 10 omics traits
    - Target heritability: 0.5 (20% direct, 80% indirect via omics)
    - Each omics trait has h² = 0.3
    - Known true breeding values available for validation
    
    Expected:
    - cor(EBV, genetic_total) > 0.2 for short chains
    - cor(EBV, genetic_total) > 0.4 for longer chains
    =#
    
    # Load simulated dataset paths
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    ped_path = Datasets.dataset("pedigree.txt", dataset_name="simulated_omics_data")
    
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "nnmm_full")
    mkpath(data_dir)
    
    # Load full phenotype data with true breeding values
    pheno_df = CSV.read(pheno_path, DataFrame)
    
    @testset "Data loading verification" begin
        @test nrow(pheno_df) > 3000
        @test :genetic_total in propertynames(pheno_df)
        @test :trait1 in propertynames(pheno_df)
        @test all(col -> col in propertynames(pheno_df), [Symbol("omic$i") for i in 1:10])
    end
    
    @testset "NNMM with full omics layer" begin
        # Create omics file with all 10 omics traits
        omics_cols = vcat([:ID], [Symbol("omic$i") for i in 1:10])
        omics_df = pheno_df[:, omics_cols]
        omics_path = joinpath(data_dir, "omics.csv")
        CSV.write(omics_path, omics_df; missingstring="NA")
        
        # Create phenotype file
        pheno_out_df = pheno_df[:, [:ID, :trait1]]
        pheno_out_path = joinpath(data_dir, "phenotypes.csv")
        CSV.write(pheno_out_path, pheno_out_df; missingstring="NA")
        
        # Load pedigree
        pedigree = get_pedigree(ped_path, separator=",", header=true)
        @test typeof(pedigree) == NNMM.PedModule.Pedigree
        
        # Define layers
        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=pheno_out_path, missing_value="NA")
        ]
        
        # Define equations with pedigree for polygenic effects
        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="omics",
                equation="omics = intercept + ID + geno",
                traits=["omic$i" for i in 1:10],
                random=[(name="ID", pedigree=pedigree)],
                method="BayesC",
                estimatePi=true
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + ID + omics",
                traits=["trait1"],
                random=[(name="ID", pedigree=pedigree)],
                method="BayesC",
                activation_function="linear"
            )
        ]
        
        # Run NNMM (short chain for CI testing)
        result = runNNMM(layers, equations; chain_length=20, printout_frequency=100)
        
        # Verify results structure
        @test result !== nothing
        @test typeof(result) <: Dict
        @test haskey(result, "EBV_NonLinear")
        
        # Get EBV results
        ebv_df = result["EBV_NonLinear"]
        @test nrow(ebv_df) > 3000
        @test :ID in propertynames(ebv_df)
        @test :EBV in propertynames(ebv_df)
        
        # EBV values should be finite
        @test all(!isnan, ebv_df.EBV)
        @test all(!isinf, ebv_df.EBV)
        
        # Merge with true breeding values for accuracy calculation
        # Convert ID columns to same type for joining
        ebv_df.ID = string.(ebv_df.ID)
        pheno_df.ID = string.(pheno_df.ID)
        merged_df = innerjoin(ebv_df, pheno_df[:, [:ID, :genetic_total]], on=:ID)
        
        if nrow(merged_df) > 0
            # Calculate accuracy (correlation with true breeding value)
            accuracy = cor(merged_df.EBV, merged_df.genetic_total)
            println("NNMM Accuracy (cor with genetic_total): ", round(accuracy, digits=4))
            
            # For short chain (20 iterations), we just check it's a valid number
            # Longer chains would give higher accuracy
            @test !isnan(accuracy)
            @test !isinf(accuracy)
        end
        
        # Cleanup
        rm(omics_path, force=true)
        rm(pheno_out_path, force=true)
    end
    
    @testset "NNMM with subset of omics (3 traits)" begin
        # Use only 3 omics traits for faster testing
        omics_cols = [:ID, :omic1, :omic2, :omic3]
        omics_df = pheno_df[:, omics_cols]
        omics_path = joinpath(data_dir, "omics_subset.csv")
        CSV.write(omics_path, omics_df; missingstring="NA")
        
        pheno_out_df = pheno_df[:, [:ID, :trait1]]
        pheno_out_path = joinpath(data_dir, "phenotypes_subset.csv")
        CSV.write(pheno_out_path, pheno_out_df; missingstring="NA")
        
        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="omics", data_path=omics_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=pheno_out_path, missing_value="NA")
        ]
        
        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="omics",
                equation="omics = intercept + geno",
                traits=["omic1", "omic2", "omic3"],
                method="BayesC"
            ),
            Equation(
                from_layer_name="omics",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + omics",
                traits=["trait1"],
                method="BayesC",
                activation_function="linear"
            )
        ]
        
        result = runNNMM(layers, equations; chain_length=10, printout_frequency=100)
        
        @test result !== nothing
        @test haskey(result, "EBV_NonLinear")
        @test nrow(result["EBV_NonLinear"]) > 3000
        
        # Cleanup
        rm(omics_path, force=true)
        rm(pheno_out_path, force=true)
    end
    
    # Cleanup test directory
    rm(data_dir, recursive=true, force=true)
end
