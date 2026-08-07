using Test

@testset "SparsifyAndSweep" begin
    @testset "kernel" begin
        include("test_kernel.jl")
    end
    @testset "stencils" begin
        include("test_stencils.jl")
    end
    @testset "system" begin
        include("test_system.jl")
    end
    @testset "boundary" begin
        include("test_boundary.jl")
    end
    @testset "sweep" begin
        include("test_sweep.jl")
    end
    @testset "solver" begin
        include("test_solver.jl")
    end
    @testset "3D" begin
        include("test_3d.jl")
    end
end
