using NoisyReach

using Test
using ControlSystemsBase
using LinearAlgebra
using ReachabilityAnalysis

# Neural network latency
const latency = 0.02
# Error of each individual dimension
const errors = [0.27, 0.27]

@testset "models" begin
    s = benchmarks[:F1]
    @test size(s.A) == (2, 2)
    @test size(s.B) == (2, 1)
    @test c2d(s, latency).A ≈ [1.0 0.12999999999999998; 0.0 1.0]
end

@testset "controllers" begin
    s = benchmarks[:F1]
    @test delay_lqr(s, latency) ≈ [0.5829779235411586 0.9271753376618778 0.3501109341069689]
    @test pole_place(s, latency) ≈ [0.19538461538461566 0.5207 0.20000000000000026]
end

@testset "reachability" begin
    s = c2d(benchmarks[:F1], latency)
    A = s.A
    B = s.B
    K = lqr(ControlSystemsBase.Discrete, A, B, I, I)
    @test size(K) == (1, 2)

    # Perception error bound zonotope
    E = Zonotope(zeros(Float64, 2), Diagonal(errors))

    # Closed-loop dynamics
    Φ = A - B * K

    # Overall error bound zonotope
    W = get_error_bound(B, K, E)
    @test W.center == [0., 0.]
    @test size(W.generators) == (2, 1)
    @test isequivalent(W, -B*K*E)

    x0center = 10.
    x0size = 1.
    x0 = Zonotope(x0center * ones(2), x0size * I(2))

    @time r = reach(Φ, x0, W, 100)
    @info max_diam(r)
    @test max_diam(r) > 0

    # Testing reachability analysis with the error bound zonotope W defined as a function
    W_func = (k, x) -> let
        # Get the vertices of the zonotope; `stack` combines the list of vectors into a matrix.
        vertices_matrix = vertices_list(x) |> stack
        # Find the maximum value for each state dimension, after taking the absolute values.
        max_states = maximum(abs.(vertices_matrix), dims=2) |> vec
        # Calculate the new error bound zonotope
        get_error_bound(B, K, Zonotope(zeros(Float64, 2), Diagonal(errors .* max_states)))
    end
    # For some reason this is really slow; I'm setting max_order to 10 for now.
    @time r = reach(Φ, x0, W_func, 100, max_order=10)
    @info max_diam(r)
    @test max_diam(r) > 0
end
