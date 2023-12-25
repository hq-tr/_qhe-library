module Potentials
include("Misc.jl")
include("FQH_state_v2.jl")
include("HaldaneSphere.jl")
include("Density.jl")
using .FQH_states
using .MiscRoutine
using .HaldaneSphere
using .ParticleDensity

using LinearAlgebra
using SparseArrays
using Arpack


function gen_onebody_element!(mat::SparseMatrixCSC{ComplexF64, Int64}, i::Int, j::Int, Lam::BitVector,Mu::BitVector, C::Matrix{T} where T<:Number, height::Float64)
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        term = height * (-1)^(a+b) * C[Lam_a+1, Mu_b+1]
        mat[i,j] += term
        mat[j,i] += conj(term)
    elseif count_difference == 0
        #println(bin2dex(Lam))
        mat[i,j] += height * sum([C[m+1,m+1] for m in bin2dex(Lam)])
    end    
end


function gen_onebody_matrix(basis_list::Vector{BitVector}, C::Matrix{T} where T<:Number,height=1.,shift=0.)
    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        #print("\r$i\t")
        for j in i:dim
            gen_onebody_element!(mat, i, j, basis_list[i], basis_list[j], C, height)
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

function gen_onebody_groundstate(basis_list::Vector{BitVector}, C::Matrix{T} where T<:Number,shift=0.;fname="")
    mat = gen_onebody_matrix(basis_list, C, 1., shift)
    E, V = eigs(mat, nev=5, sigma=0)
    println("Energy eigenvalues:")
    display(E)
    gs = FQH_state(basis_list, V[:,1])
    if !isempty(fname) printwf(gs, fname=fname) end
    return gs    
end

function diracdelta_element!(mat::SparseMatrixCSC{ComplexF64, Int64}, i::Int, j::Int, Lam::BitVector,Mu::BitVector, pos::Number, height::Number)
    R = abs(pos)
    θ = angle(pos) + π
    check_difference = Lam .⊻ Mu
    count_difference = count(check_difference)
    if count_difference == 2
        Lam_a = bin2dex(check_difference.*Lam)[1]
        Mu_b  = bin2dex(check_difference.*Mu)[1]
        a = count(Lam[1:Lam_a])
        b = count(Mu[1:Mu_b])
        term = height * (-1)^(a+b) * (-1)^(Mu_b-Lam_a) * π * R^(Lam_a+Mu_b) * exp(-R^2/2) * exp(im*(Mu_b-Lam_a)*θ)/((√2)^(Lam_a+Mu_b)*sqfactorial(Lam_a)*sqfactorial(Mu_b))
        mat[i,j] += term
        if i!=j mat[j,i] += conj(term) end
    elseif count_difference == 0
        #println(bin2dex(Lam))
        for m in bin2dex(Lam)
            mat[i,j] += height *  π*R^(2m) * exp(-R^2/2)/(2^m*factorial(big(m)))
        end
    end    
end

function diracdelta_matrix(basis_list::Vector{BitVector}, pos::Number,height=1.,shift=0.)
    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        #print("\r$i\t")
        for j in i:dim
            diracdelta_element!(mat, i, j, basis_list[i], basis_list[j], pos, height)
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

function diracdelta_matrix(basis_list::Vector{BitVector}, pos::Vector{T} where T<:Number,height=1.,shift=0.)
    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        #print("\r$i\t")
        for j in i:dim
            for po in pos
                diracdelta_element!(mat, i, j, basis_list[i], basis_list[j], po,height)
            end
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

function diracdelta_groundstate(basis_list::Vector{BitVector}, pos::Number,shift=0.;fname="")
    mat = diracdelta_matrix(basis_list, pos, shift)
    E, V = eigs(mat, nev=5, sigma=0)
    println("Energy eigenvalues:")
    display(E)
    gs = FQH_state(basis_list, V[:,1])
    if !isempty(fname) printwf(gs, fname=fname) end
    return gs    
end

function diracdelta_groundstate(basis_list::Vector{BitVector}, pos::Vector{T} where T<:Number, shift=0.;fname="groundstate")
    # More than one trap
    dim = length(basis_list)
    mat = spzeros(Complex{Float64}, (dim,dim))
    for p in pos
        mat += diracdelta_matrix(basis_list, p)
    end
    if shift!=0
        mat += shift * sparse(I,dim,dim)
    end
    E, V = eigs(mat, nev=5, sigma=0)
    println("Energy eigenvalues:")
    display(E)
    gs = FQH_state(basis_list, V[:,1])
    if !isempty(fname) printwf(gs; fname=fname) end
    return gs    
end

function sphere_bump_matrix(basis_list::Vector{BitVector},θ::Float64, ϕ::Float64, height=1.0,shift=0.0)
    No = length(basis_list[1])
    coef = one_particle_state(θ,ϕ,No-1).coef
    C  = coef * coef'

    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))
    for i in 1:dim
        #print("\r$i\t")
        for j in i:dim
            gen_onebody_element!(mat, i, j, basis_list[i], basis_list[j], C, height)
        end
    end
    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

function sphere_bump_matrix(basis_list::Vector{BitVector},θ::Vector{Float64}, ϕ::Vector{Float64}, height=1.0,shift=0.0)
    No = length(basis_list[1])

    dim = length(basis_list)
    mat = spzeros(Complex{Float64},(dim,dim))

    for k in length(θ)
        coef = one_particle_state(θ[k],ϕ[k],No-1).coef
        C  = coef * coef'
        for i in 1:dim
            #print("\r$i\t")
            for j in i:dim
                gen_onebody_element!(mat, i, j, basis_list[i], basis_list[j], C, height)
            end
        end
    end

    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 

function sphere_point_matrix(basis_list::Vector{BitVector},θ::Float64, ϕ::Float64, height=1.0,shift=0.0)
    No = length(basis_list[1])

    S = (No-1.0)/2.0
    sfunction = map(m->single_particle_state_sphere(π-θ,ϕ,S,m), -S:1:S)
    
    mat = height.*density_matrix(basis_list, sfunction)

    if shift!=0 mat += shift * sparse(I, dim, dim) end
    return mat
end 




export diracdelta_matrix, diracdelta_groundstate, diracdelta_element!, gen_onebody_matrix, gen_onebody_element!, gen_onebody_groundstate, sphere_bump_matrix, sphere_point_matrix

end