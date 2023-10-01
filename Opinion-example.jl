## Simulation of opinion dynamics model with bias
## written by : Christian Koertje


using ProgressBars

## interaction kernel function (perception kernel)
function g(y::Float64, μ::Float64, σ::Float64, b::Float64)::Float64
    return exp(-((y - μ) / σ)^2) + (b - 1)  * exp(-((y + μ) / σ)^2)
end

## initialization 
function initialize()
    global P = 1 .+ 0.02 * randn(N)
end

## executing a time step 
function update()
    global P
    @fastmath @inbounds begin
        # get local neighbors
        lP = circshift(P, 1)
        rP = circshift(P, -1)

        # compute spatial derivatives
        ∇²P = rP .+ lP .- 2*P   # laplacian
        ∇P = rP .- lP           # gradient

        # nonlocal terms ∇⋅(P * G) = G ∇P + P ∇G
        G = Vector{Vector{Float64}}()
        ∇G = Vector{Vector{Float64}}()
        for dy ∈ -r:r
            y = dy * Δx
            W = g(y, μ, σ, b)
            push!(G,  circshift(P,  -dy) * W)
            push!(∇G, circshift(∇P, -dy) * W)
        end

        G∇P = sum(G) .* ∇P
        P∇G = P .* sum(∇G)

        # update via forward-euler
        nextP = @. P + (Δt * d / Δx^2) * ∇²P - (Δt * c / (2*Δx)) * G∇P - (Δt * c / (2*Δx)) * P∇G
        P, nextP = nextP, P
    end
end


function main()
    print("Begin simulation...\n")
    ## parameters
    global N = 1000                 # number of cells
    global L = floor(10 * 2π)       # length of domain
    global Δx = L / N               # spatial resolution
    global R = L / 2                # radius of nonlocal term
    global r = floor(R / Δx)        # resolution of nonlocal term
    global Δt = 0.001               # temporal resolution
    global T = 5.                   # simulation time

    ## model
    global d = 0.05                 # diffusion constant
    global c = 0.1                  # rate of migration

    ## kernel
    global μ, σ = 1.0, 1.0          # offset and width of information gathering
    global b = 0.0                  # bias b ∈ [0,2)

    # execution
    initialize()
    for t ∈ ProgressBar(Δt:Δt:T) 
        update()
    end    
    print("Simulation complete!\n")
end

main()