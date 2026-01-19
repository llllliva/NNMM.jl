using Test
using NNMM
using NNMM.Datasets
using DataFrames
using CSV
using Statistics
using Random

@testset "NNMM as Traditional BayesC" begin
    #=
    This test demonstrates how NNMM generalizes traditional genomic prediction models.
    
    By using:
    - Latent nodes in the middle layer that are COMPLETELY MISSING for all individuals
    - A LINEAR activation function
    
    The NNMM framework can be used for genomic prediction:
    
        Genotypes ──► [Missing Latent Nodes] ──► Phenotype
                          (sampled)              (linear)
    
    This is mathematically equivalent to traditional BayesC.
    
    Using simulated_omics_data for validation with known true breeding values.
    =#
    
    # Use simulated omics dataset
    geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
    pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")
    
    # Setup test data directory
    data_dir = joinpath(@__DIR__, "fixtures", "bayesc_pattern")
    mkpath(data_dir)
    
    # Load phenotype data (includes true genetic values)
    pheno_df = CSV.read(pheno_path, DataFrame)
    nind = nrow(pheno_df)
    
    @testset "BayesC via NNMM with latent layer" begin
        # Create middle layer with completely missing latent nodes
        # Note: Using 2 latent nodes to avoid scalar matrix issue in HMC
        omics_df = DataFrame(
            ID = pheno_df.ID,
            latent1 = fill(missing, nind),
            latent2 = fill(missing, nind)
        )
        o_path = joinpath(data_dir, "latent_missing.csv")
        CSV.write(o_path, omics_df; missingstring="NA")
        
        # Verify the file has all missing values
        omics_check = CSV.read(o_path, DataFrame; missingstring="NA")
        @test all(ismissing, omics_check.latent1)
        @test all(ismissing, omics_check.latent2)
        
        # Create phenotype file
        pheno_out_df = pheno_df[:, [:ID, :trait1]]
        y_path = joinpath(data_dir, "phenotypes.csv")
        CSV.write(y_path, pheno_out_df; missingstring="NA")
        
        # Define 3-layer network
        layers = [
            Layer(layer_name="geno", data_path=[geno_path]),
            Layer(layer_name="latent", data_path=o_path, missing_value="NA"),
            Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
        ]
        
        # Define equations:
        # 1. Genotypes → Latent (BayesC marker regression)
        # 2. Latent → Phenotype (LINEAR activation)
        equations = [
            Equation(
                from_layer_name="geno",
                to_layer_name="latent",
                equation="latent = intercept + geno",
                traits=["latent1", "latent2"],
                method="BayesC",
                estimatePi=true
            ),
            Equation(
                from_layer_name="latent",
                to_layer_name="phenotypes",
                equation="phenotypes = intercept + latent",
                traits=["trait1"],
                method="BayesC",
                activation_function="linear"  # Key: linear activation
            )
        ]
        
        # Verify configuration
        @test equations[1].method == "BayesC"
        @test equations[2].activation_function == "linear"
        
        # Run NNMM
        result = runNNMM(layers, equations; chain_length=10, printout_frequency=100)
        
        # Verify results
        @test result !== nothing
        @test typeof(result) <: Dict
        @test haskey(result, "EBV_NonLinear")
        
        ebv_df = result["EBV_NonLinear"]
        @test nrow(ebv_df) > 3000
        @test :ID in propertynames(ebv_df)
        @test :EBV in propertynames(ebv_df)
        
        # EBV values should be finite
        @test all(!isnan, ebv_df.EBV)
        @test all(!isinf, ebv_df.EBV)
        
        # Calculate accuracy against true genetic values
        ebv_df.ID = string.(ebv_df.ID)
        pheno_df.ID = string.(pheno_df.ID)
        merged_df = innerjoin(ebv_df, pheno_df[:, [:ID, :genetic_total]], on=:ID)
        
        if nrow(merged_df) > 0
            accuracy = cor(merged_df.EBV, merged_df.genetic_total)
            println("BayesC via NNMM Accuracy: ", round(accuracy, digits=4))
            @test !isnan(accuracy)
        end
    end
    
    @testset "Different Bayesian methods via NNMM" begin
        # Test that different methods work with the latent layer approach
        o_path = joinpath(data_dir, "latent_missing.csv")
        y_path = joinpath(data_dir, "phenotypes.csv")
        
        for method in ["BayesC", "BayesA"]
            @testset "Method: $method" begin
                layers = [
                    Layer(layer_name="geno", data_path=[geno_path]),
                    Layer(layer_name="latent", data_path=o_path, missing_value="NA"),
                    Layer(layer_name="phenotypes", data_path=y_path, missing_value="NA")
                ]
                
                estimate_pi = method in ["BayesB", "BayesC"]
                
                equations = [
                    Equation(
                        from_layer_name="geno",
                        to_layer_name="latent",
                        equation="latent = intercept + geno",
                        traits=["latent1", "latent2"],
                        method=method,
                        estimatePi=estimate_pi
                    ),
                    Equation(
                        from_layer_name="latent",
                        to_layer_name="phenotypes",
                        equation="phenotypes = intercept + latent",
                        traits=["trait1"],
                        method=method,
                        activation_function="linear"
                    )
                ]
                
                result = runNNMM(layers, equations; chain_length=5, printout_frequency=100)
                
                @test result !== nothing
                @test haskey(result, "EBV_NonLinear")
                @test nrow(result["EBV_NonLinear"]) > 3000
            end
        end
    end
    
    # Cleanup
    rm(data_dir, recursive=true, force=true)
end
