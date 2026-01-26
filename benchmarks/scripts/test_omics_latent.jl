using JWAS, NNMM, NNMM.Datasets, DataFrames, CSV, Random, Statistics

Random.seed!(12345)

println("="^70)
println("TEST 4: NNMM (Latent + Observed Omics) vs Traditional")
println("="^70)
flush(stdout)

# Use existing dataset but subset for speed
geno_path = Datasets.dataset("genotypes_1000snps.txt", dataset_name="simulated_omics_data")
pheno_path = Datasets.dataset("phenotypes_sim.txt", dataset_name="simulated_omics_data")

geno_df = CSV.read(geno_path, DataFrame)
pheno_df = CSV.read(pheno_path, DataFrame)

# Subset to 500 individuals for speed
n_subset = 500
subset_idx = 1:n_subset
geno_df = geno_df[subset_idx, :]
pheno_df = pheno_df[subset_idx, :]
n = nrow(pheno_df)

geno_mat = Matrix{Float64}(geno_df[:, 2:end])
n_markers = size(geno_mat, 2)

# Simulate 1 observed omics
β_omics = randn(n_markers) * 0.02
omics1 = geno_mat * β_omics + randn(n) * 0.5

# Simulate direct genetic effect
β_direct = randn(n_markers) * 0.02
genetic_direct = geno_mat * β_direct

# Phenotype
y = genetic_direct + 0.5 * omics1 + randn(n) * 0.3

println("Data: n=$n (subset), markers=$n_markers")
println("Model: y = genetic_direct + 0.5*omics + noise")
println()
flush(stdout)

tmpdir = mktempdir()

# Save subset genotype file
geno_df.ID = string.(geno_df.ID)
geno_subset_path = joinpath(tmpdir, "geno_subset.csv")
CSV.write(geno_subset_path, geno_df)

# ============================================================
# MODEL 1: JWAS NNMM (equivalent structure)
# ============================================================
println("MODEL 1: JWAS NNMM (latent_traits=[latent1(missing), omics1(observed)])")
flush(stdout)

# Phenotype data with omics as a latent trait column
# latent1 must be Union{Missing, Float64} type for JWAS to assign values
jwas_pheno = DataFrame(
    ID = geno_df.ID, 
    y = y, 
    latent1 = Vector{Union{Missing, Float64}}(fill(missing, n)),
    omics1 = omics1
)

# Load genotypes first - must assign to 'geno' variable for JWAS NNMM
geno = JWAS.get_genotypes(geno_subset_path)

# Build NNMM model: include 'geno' in equation, 2 hidden nodes
model1 = JWAS.build_model("y = intercept + geno",
                          num_hidden_nodes=2,
                          nonlinear_function="linear",
                          latent_traits=["latent1", "omics1"])

println("Running JWAS NNMM...")
flush(stdout)
out1 = JWAS.runMCMC(model1, jwas_pheno, 
                    chain_length=3000, burnin=1000,
                    printout_frequency=500,
                    output_folder=joinpath(tmpdir, "jwas"))

println("JWAS completed!")
flush(stdout)

# ============================================================
# MODEL 2: NNMM
# ============================================================
println()
println("MODEL 2: NNMM (Layer2 = [latent1, omics1])")
flush(stdout)

layer2_df = DataFrame(ID = geno_df.ID, latent1 = fill(missing, n), omics1 = omics1)
CSV.write(joinpath(tmpdir, "layer2.csv"), layer2_df; missingstring="NA")

nnmm_pheno = DataFrame(ID = geno_df.ID, y = y)
CSV.write(joinpath(tmpdir, "pheno.csv"), nnmm_pheno)

layers = [
    Layer(layer_name="geno", data_path=[geno_subset_path]),
    Layer(layer_name="middle", data_path=joinpath(tmpdir, "layer2.csv"), missing_value="NA"),
    Layer(layer_name="pheno", data_path=joinpath(tmpdir, "pheno.csv"), missing_value="NA")
]

equations = [
    Equation(from_layer_name="geno", to_layer_name="middle",
             equation="middle = intercept + geno", traits=["latent1"], method="BayesC"),
    Equation(from_layer_name="middle", to_layer_name="pheno",
             equation="pheno = intercept + middle", traits=["y"], activation_function="linear")
]

println("Running NNMM MCMC...")
flush(stdout)
out2 = runNNMM(layers, equations; chain_length=3000, burnin=1000,
               printout_frequency=500, output_folder=joinpath(tmpdir, "nnmm"))

println("NNMM completed!")
flush(stdout)

# ============================================================
# COMPARE
# ============================================================
println()
println("="^70)
println("RESULTS")
println("="^70)

# Check available keys
println("JWAS output keys: ", keys(out1))
println("NNMM output keys: ", keys(out2))
println()

# JWAS NNMM uses EBV_NonLinear for the final prediction
ebv_jwas = out1["EBV_NonLinear"]; ebv_jwas.ID = string.(ebv_jwas.ID)
epv_nnmm = out2["EBV_NonLinear"]; epv_nnmm.ID = string.(epv_nnmm.ID)

merged = innerjoin(rename(ebv_jwas, :EBV=>:JWAS), rename(epv_nnmm[:, [:ID,:EBV]], :EBV=>:NNMM), on=:ID)
cor_result = cor(Float64.(merged.JWAS), Float64.(merged.NNMM))

true_genetic = genetic_direct + 0.5 * omics1
acc_jwas = cor(Float64.(ebv_jwas.EBV), true_genetic)
acc_nnmm = cor(Float64.(epv_nnmm.EBV), true_genetic)

println("r(JWAS_NNMM, NNMM.jl) = $(round(cor_result, digits=4))")
println("Accuracy: JWAS=$(round(acc_jwas, digits=4)), NNMM=$(round(acc_nnmm, digits=4))")
if cor_result > 0.95
    println("✓ SUCCESS: JWAS NNMM ≈ NNMM.jl!")
else
    println("⚠ Results differ")
end
println("="^70)
println("TEST COMPLETED!")
