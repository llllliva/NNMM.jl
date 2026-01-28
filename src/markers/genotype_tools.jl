"""
    get_column_ref(X::Vector{T})

    To obtain a vector of views (alias/pointer) for each column of the input matrix WITHOUT COPYING the underlying data. 
    input:  a matrix X
    output: a vector containing views of each column of the input matrix X
"""
function get_column_ref(X::AbstractArray{T,2}) where T <: Any
      return [view(X, :, i) for i in 1:size(X, 2)] # Create a vector of views for each column of the matrix X
end

"""
    This function centers columns of the input matrix X by subtracting their means along each column. The function operates in-place by modifying the original matrix X.

    Input:
    - X::AbstractMatrix: a matrix to be centered

    Output:
    - col_means::Vector: a vector of mean values for each column in the original matrix, computed before centering.
"""
function center!(X::AbstractArray{T,2}) where T <: Any
    col_means = mean(X, dims=1) # Calculate the means in each column
    X .-= col_means  # Subtract column means from X, 
                     # Note: in-place, using broadcasting, i.e. X[i,j] -= col_means[j]
    return col_means # Return a row vector of column means (size 1 by ncol)
end

function getXpRinvX(X, Rinv)
    XpRinvX = [((X[:,i].*Rinv)'X[:,i]) for i=1:size(X,2)]
    return XpRinvX
end

function getXpRinvX(X)
    XpRinvX = [dot(X[:,i],X[:,i]) for i=1:size(X,2)]
    return XpRinvX
end

function get_column_blocks_ref(X,fast_blocks=false)
    xArray = Array{typeof(X)}(undef,length(fast_blocks))
    @showprogress "building prerequisite matrices ..." for i in 1:length(fast_blocks)
        pos_start = fast_blocks[i]
        pos_end   = (i != length(fast_blocks)) ? (fast_blocks[i+1]-1) : size(X,2)
        xArray[i] = view(X,:,pos_start:pos_end)
    end
    return xArray
end

mutable struct GibbsMats
    X::Union{Array{Float64,2},Array{Float32,2}}
    nrows::Int64
    ncols::Int64
    xArray     #do not declare type because it is a vector of views
    xRinvArray #do not declare type because it is a vector of views
    xpRinvx::Union{Array{Float64,1},Array{Float32,1}}
    XArray
    XRinvArray
    XpRinvX
    function GibbsMats(X::Union{Array{Float64,2},Array{Float32,2}},Rinv;fast_blocks=false)
        #single-locus, adjusting y, approach
        nrows,ncols = size(X)
        xArray      = get_column_ref(X)
        xpRinvx     = getXpRinvX(X,Rinv)
        if Rinv == ones(length(Rinv))
            xRinvArray = xArray  #avoid using extra memory for xRinvArray
        else
            xRinvArray = [x.*Rinv for x in xArray]
        end
        #block, adjusting rhs and/or y, approach
        if fast_blocks != false
            XArray = get_column_blocks_ref(X,fast_blocks)
            XRinvArray = @showprogress "building prerequisite matrices ..." [X'Diagonal(Rinv) for X in XArray]
            XpRinvX = @showprogress "building prerequisite matrices ..." [XRinvArray[i]*XArray[i] for i in 1:length(XArray)]
        else
            XArray = XRinvArray = XpRinvX = false
        end
        new(X,nrows,ncols,
            xArray,xRinvArray,xpRinvx,
            XArray,XRinvArray,XpRinvX)
    end
end

################################################################################
#align genotypes with phenotyps given IDs #MAY USE TOO MUCH MEMORY
################################################################################
#input: mme.M.genotypes (1) complete genomic data  -> all genotyped individuals
#                       (2)* incomplete genomic data -> all pedigree individuals
#output:mme.M.genotypes (1) subset
#
# mme.M.genotypes is modified to same IDs in phenotypes for coding convenience
#
#Note that Aligning genotypes with phenotypes and output_ID in single-step analysis
#(incomplete genomic data) has been moved to SSBR.jl file.
function align_genotypes(mme::MME,output_heritability=false,single_step_analysis=false)
    if single_step_analysis==false
        if mme.output_ID != 0
            for Mi in mme.M
                if !(Mi isa Genotypes)
                    continue
                end
                if mme.output_ID != Mi.obsID
                    Zo  = map(Float32,mkmat_incidence_factor(mme.output_ID,Mi.obsID))
                    Mi.output_genotypes =  Zo*Mi.genotypes
                else
                    Mi.output_genotypes =  Mi.genotypes #reference, not copy
                end
                if Mi.isGRM #relationship matrix is provided
                    Z  = map(Float32,mkmat_incidence_factor(mme.obsID,Mi.obsID))
                    Mi.output_genotypes = Mi.output_genotypes*Z'
                end
            end
        end
        #***************************************************************************
        #Align genotypes with phenotypes
        #Change mme.M.genotypes to phenotyped individuals to accommodate
        #individuals with repeated records or individuals without records
        #
        #**********CENTERING?*******************************************************
        if mme.obsID != mme.M[1].obsID
            for Mi in mme.M
                if !(Mi isa Genotypes)
                    continue
                end
                Z  = map(Float32,mkmat_incidence_factor(mme.obsID,Mi.obsID))
                genotypes = Z*Mi.genotypes
                if Mi.isGRM #relationship matrix is provided
                    genotypes = genotypes*Z'
                end
                Mi.genotypes = genotypes
                Mi.obsID     = mme.obsID
                Mi.nObs      = length(mme.obsID)
            end
        end
    end
end

#compared to the above align_genotypes() function:
#Mi.genotypes -> Mi.data
#remove: Align genotypes with phenotypes; it should be used in each iteration of MCMC because omics is updated at each iteration
function align_omics(mme::MME,output_heritability=false,single_step_analysis=false)
    if single_step_analysis==false
        if mme.output_ID != 0
            for Mi in mme.M
                if !(Mi isa Omics)
                    continue
                end
                omicsID=Mi.featureID
                Mi_genotypes = Matrix(Mi.data[!,omicsID]) #only select the omics columns (i.e., covariates/random effect terms are not included)
                if mme.output_ID != Mi.obsID
                    Zo  = map(Float32,mkmat_incidence_factor(mme.output_ID,Mi.obsID))
                    Mi.output_genotypes =  Zo*Mi_genotypes
                else
                    Mi.output_genotypes =  Mi_genotypes #reference, not copy
                end
                if Mi.isGRM #relationship matrix is provided
                    Z  = map(Float32,mkmat_incidence_factor(mme.obsID,Mi.obsID))
                    Mi.output_genotypes = Mi.output_genotypes*Z'
                end
            end
        end
    end
end


#need to align after omics is updated at each iteration 
#full omics is used for 1->2 (including testing inds without yobs)
#partial omics is used for 2->3 (only for training inds with yobs)
function align_transformed_omics_with_phenotypes(mme::MME,nonlinear_function)
        #***************************************************************************
        #Align genotypes with phenotypes
        #Change mme.M.genotypes to phenotyped individuals to accommodate
        #individuals with repeated records or individuals without records
        #
        #**********CENTERING?*******************************************************
    if mme.M != 0 && length(mme.M) > 0 && mme.obsID != mme.M[1].obsID
        for Mi in mme.M
            if !(Mi isa Omics)
                continue
            end
            Z  = map(Float32,mkmat_incidence_factor(mme.obsID,Mi.obsID))
            omicsID=Mi.featureID
            Mi_omics = Matrix(Mi.data[!,omicsID]) #only select the omics columns (i.e., covariates/random effect terms are not included)
            Mi.aligned_omics_w_phenotype = Z*Mi_omics
            Mi.aligned_omics_w_phenotype = nonlinear_function.(Mi.aligned_omics_w_phenotype)
            Mi.aligned_obsID_w_phenotype = mme.obsID
            Mi.aligned_nObs_w_phenotype = length(mme.obsID)
        end
    else
        for Mi in mme.M
            if !(Mi isa Omics)
                continue
            end
            omicsID=Mi.featureID
            Mi_omics = Matrix(Mi.data[!,omicsID]) #only select the omics columns (i.e., covariates/random effect terms are not included)
            Mi.aligned_omics_w_phenotype = nonlinear_function.(Mi_omics)
            Mi.aligned_obsID_w_phenotype = mme.obsID
            Mi.aligned_nObs_w_phenotype = length(mme.obsID)
        end
    end
end


"""
mkmat_incidence_factor(yID::Vector, uID::Vector)
    create an incidence matrix Z to reorder uID to yID by yID = Z*uID.
    input: 
        - yID: a vector containing the desired order of IDs
        - uID: a vector containing the original order of IDs
    output: 
        - Z: a sparse matrix representing the incidence relationship between yID and uID (yID = Z*uID)
"""
function mkmat_incidence_factor(yID::Vector, uID::Vector)
    Z = spzeros(length(yID), length(uID)) # initialize a sparse matrix Z with the dimensions of yID and uID
    uIDdict = Dict(id => index for (index, id) in enumerate(uID))  # create a uID dictionary to map uID elements to their indices
    
    # iterate over yID elements to populate Z matrix
    for (rowi, id) in enumerate(yID)
        if haskey(uIDdict, id) # check if the current yID element exists in uID dictionary
            index = uIDdict[id]
        else
            error("$id is not found!") # error if the id is not found in the uID dictionary
        end
        Z[rowi, index] = 1 # set the corresponding element in the Z matrix to 1
    end
    return Z
end

################################################################################
#Set Marker Effcets Hyperparameters: Variances and Pi
################################################################################
function set_marker_hyperparameters_variances_and_pi(mme::MME)
    if mme.M != 0
        for Mi in mme.M
            #(1) Pi in multi-trait analysis
            if (mme.nModels !=1 || mme.MCMCinfo.RRM != false) && Mi.π==0.0 && !(Mi.method in ["RR-BLUP","BayesL"])
                println()
                printstyled("Pi (Π) is not provided.\n",bold=false)
                printstyled("Pi (Π) is generated assuming all markers have effects on all traits.\n",bold=false)
                mykey=Array{Float64}(undef,0)
                ntraits=mme.nModels
                if mme.MCMCinfo.RRM != false
                    ntraits = size(mme.MCMCinfo.RRM,2)
                end
                Pi=Dict{Array{Float64,1},Float64}()
                #for i in [ bin(n,ntraits) for n in 0:2^ntraits-1 ] `bin(n, pad)` is deprecated, use `string(n, base=2, pad=pad)
                for i in [ string(n,base=2,pad=ntraits) for n in 0:2^ntraits-1 ]
                    Pi[parse.(Float64, split(i,""))]=0.0
                end
                Pi[ones(ntraits)]=1.0
                Mi.π = Pi
            end
            #(2) marker effect variances
            if Mi.G.val == false
                if Mi.method!="GBLUP"
                    genetic2marker(Mi,Mi.π)
                    println()
                    if mme.nModels != 1 || mme.MCMCinfo.RRM != false
                      if !isposdef(Mi.G.val) #also work for scalar
                        error("Marker effects covariance matrix is not postive definite! Please modify the argument: Pi.")
                      end
                      println("The prior for marker effects covariance matrix is calculated from genetic covariance matrix and Π.")
                      if mme.MCMCinfo.printout_model_info==true
                          if Mi.G.constraint == true
                              # For constraint=true (independent traits), print only diagonal values
                              println("The mean of the prior for the marker effects variance (diagonal):")
                              max_traits = 5
                              for t in 1:min(Mi.ntraits, max_traits)
                                  @printf("  trait %d: %.6f\n", t, Mi.G.val[t,t])
                              end
                              if Mi.ntraits > max_traits
                                  println("  ... (", Mi.ntraits - max_traits, " more traits)")
                              end
                          else
                              println("The mean of the prior for the marker effects covariance matrix is:")
                              print_matrix_truncated(Mi.G.val; digits=6)
                          end
                      end
                    else
                      if !isposdef(Mi.G.val) #positive scalar (>0)
                        error("Marker effects variance is negative!")
                      end
                      println("The prior for marker effects variance is calculated from the genetic variance and π.")
                      print("The mean of the prior for the marker effects variance is: ")
                      print(round.(Mi.G.val,digits=6))
                    end
                    print("\n\n\n")
                elseif Mi.method == "GBLUP"
                    Mi.G.val  = Mi.genetic_variance.val
                end
            end
            #(3) scale parameter for marker effect variance
            # The scale formula depends on the sampling distribution:
            # - Single-trait or diagonal-only (constraint=true): Scaled Inverse Chi-square
            #   E[Scale-Inv-χ²(ν, s²)] = νs²/(ν-2), so scale = prior_mean * (df-2) / df
            # - Full covariance (constraint=false): Inverse Wishart  
            #   E[IW(ν, S)] = S/(ν-p-1), so scale = prior_mean * (df-ntraits-1)
            if Mi.ntraits == 1 && mme.MCMCinfo.RRM == false
                # Single trait: Scale-Inv-χ² with df > 2
                if Mi.G.df <= 2
                    @warn "Degrees of freedom ($(Mi.G.df)) should be > 2 for valid Scale-Inv-χ² prior"
                end
                Mi.G.scale = Mi.G.val*(Mi.G.df-Mi.ntraits-1)/Mi.G.df
            elseif Mi.G.constraint == true
                # Multi-trait with diagonal-only (independent variances): Scale-Inv-χ² for each
                # Each diagonal element is sampled independently, so use single-trait formula
                if Mi.G.df <= 2
                    @warn "Degrees of freedom ($(Mi.G.df)) should be > 2 for valid Scale-Inv-χ² prior"
                end
                Mi.G.scale = Mi.G.val*(Mi.G.df-2)/Mi.G.df
            else
                # Multi-trait with full covariance: Inverse Wishart
                if Mi.G.df <= Mi.ntraits + 1
                    @warn "Degrees of freedom ($(Mi.G.df)) should be > ntraits+1 ($(Mi.ntraits+1)) for valid Inverse-Wishart prior"
                end
                Mi.G.scale = Mi.G.val*(Mi.G.df-Mi.ntraits-1)
            end
        end
    end
end

function genetic2marker(M::Genotypes,Pi::Dict)
  ntraits = size(M.genetic_variance.val,1)
  denom   = zeros(ntraits,ntraits)
  for i in 1:ntraits
    for j in i:ntraits
      pi_selected = filter(d->d.first[i]==1.0 && d.first[j]==1.0,Pi)

      denom[i,j] = M.sum2pq*sum(values(pi_selected))
      denom[j,i] = denom[i,j]
    end
  end
  M.G.val = M.genetic_variance.val ./ denom
end

function genetic2marker(M::Genotypes, π::Real)
    πf = Float64(π)
    M.G.val = M.genetic_variance.val / ((1 - πf) * M.sum2pq)
end

function genetic2marker(M::Omics, π::Real)
    πf = Float64(π)
    # Omics features are general continuous covariates; unlike genotype markers (0/1/2),
    # there is no closed-form `sum2pq` based on allele frequencies. To make the prior
    # invariant to feature scaling, use the sum of per-feature variances as the
    # analogue of `sum2pq`:
    #   Var(Xa) ≈ (1-π) * σ_a² * Σ_j Var(X_j)
    #   ⇒ σ_a² ≈ σ_g² / ((1-π) * Σ_j Var(X_j))
    sumvar = 0.0
    if M.data isa DataFrame
        # Prefer column-wise accumulation to avoid materializing a dense matrix.
        for fid in M.featureID
            col = M.data[!, fid]
            v = filter(isfinite, skipmissing(col))
            if length(v) >= 2
                sumvar += var(v; corrected=true)
            end
        end
    else
        # Fallback for matrix-like omics storage (no missings expected).
        X = Matrix(M.data)
        sumvar = float(sum(var(X; dims=1)))
    end
    denom = (1 - πf) * max(sumvar, eps(Float64))
    denom = max(denom, eps(Float64))
    M.G.val = M.genetic_variance.val / denom
end
