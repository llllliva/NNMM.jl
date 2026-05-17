using Test
using NNMM

@testset "output_result marker and omic columns" begin
    function make_marker_block(name, ids, effects, effects2, model_frequency)
        marker = NNMM.Genotypes(
            AbstractString["id1"],
            string.(ids),
            1,
            length(ids),
            fill(0.5, length(ids)),
            0.5,
            true,
            zeros(1, length(ids)),
            false,
        )
        marker.name = name
        marker.trait_names = ["trait1"]
        marker.ntraits = 1
        marker.meanAlpha = [effects]
        marker.meanAlpha2 = [effects2]
        marker.meanDelta = [model_frequency]
        marker.estimatePi = false
        marker.G = NNMM.Variance(false, false, false, true, false, false)
        return marker
    end

    function make_omics_block(name, ids, effects, effects2, model_frequency)
        omics = NNMM.Omics(
            AbstractString["id1"],
            string.(ids),
            1,
            length(ids),
            zeros(1, length(ids)),
        )
        omics.name = name
        omics.trait_names = ["trait1"]
        omics.ntraits = 1
        omics.meanAlpha = [effects]
        omics.meanAlpha2 = [effects2]
        omics.meanDelta = [model_frequency]
        omics.estimatePi = false
        omics.G = NNMM.Variance(false, false, false, true, false, false)
        return omics
    end

    term = NNMM.ModelTerm("intercept", 1, "trait1")
    term.names = ["intercept"]
    model = NNMM.MME(
        1,
        AbstractString["trait1 = intercept"],
        NNMM.ModelTerm[term],
        Dict{AbstractString,NNMM.ModelTerm}("trait1:intercept" => term),
        [:trait1],
        NNMM.Variance(1.0, false, false, true, false, false),
    )
    model.MCMCinfo = NNMM.MCMCinfo(
        false, 10, 0, 1, 1, false, 10, false, false, false,
        false, false, false, false, 1234, true, "unused", false, false,
    )
    model.M = [
        make_marker_block("geno", ["snp1", "snp2"], [0.1, -0.2], [0.05, 0.08], [1.0, 0.5]),
        make_omics_block("omics", ["omic1", "omic2"], [0.3, 0.4], [0.12, 0.22], [0.75, 0.25]),
    ]

    result = NNMM.output_result(
        model,
        "unused",
        [1.25],
        2.0,
        missing,
        [1.75],
        5.0,
        missing,
    )

    expected_columns = [:Trait, :Marker_ID, :Estimate, :SD, :Model_Frequency]

    @test haskey(result, "marker effects geno")
    @test haskey(result, "marker effects omics")
    @test propertynames(result["marker effects geno"]) == expected_columns
    @test propertynames(result["marker effects omics"]) == expected_columns
    @test result["marker effects geno"].Trait == ["trait1", "trait1"]
    @test result["marker effects geno"].Marker_ID == ["snp1", "snp2"]
    @test result["marker effects omics"].Trait == ["trait1", "trait1"]
    @test result["marker effects omics"].Marker_ID == ["omic1", "omic2"]
end
