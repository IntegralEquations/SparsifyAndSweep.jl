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
    @testset "L-sweeps" begin
        include("test_lsweep.jl")
    end
    if Base.find_package("MPI") !== nothing
        @testset "L-sweeps under MPI" begin
            include("mpi_launch_test.jl")
        end
    else
        @info "MPI.jl not in this environment; skipping the distributed L-sweeps test " *
              "(run the tests from the test/ environment to include it)"
    end
end
