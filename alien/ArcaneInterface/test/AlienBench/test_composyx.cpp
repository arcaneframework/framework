
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <sstream>
#include <composyx.hpp>
#include <composyx/dist/MPI.hpp>
#include <composyx/part_data/PartMatrix.hpp>
#include <composyx/solver/BiCGSTAB.hpp>
#include <composyx/solver/GMRES.hpp>
#include <composyx/solver/Jacobi.hpp>
#if defined(COMPOSYX_USE_QRMUMPS)
#include <composyx/solver/QrMumps.hpp>
#endif
#if defined(COMPOSYX_USE_MUMPS)
#include <composyx/solver/Mumps.hpp>
#endif
#if defined(COMPOSYX_USE_PASTIX)
#include <composyx/solver/Pastix.hpp>
#endif

#if defined(COMPOSYX_USE_EIGEN)
#include <composyx/wrappers/Eigen/Eigen_header.hpp>
#include <composyx/wrappers/Eigen/Eigen.hpp>
#include <composyx/wrappers/Eigen/EigenSparseSolver.hpp>
#endif
#include <composyx/precond/DiagonalPrecond.hpp>
#include <composyx/precond/AbstractSchwarz.hpp>
#include <composyx/precond/TwoLevelAbstractSchwarz.hpp>
#include <composyx/precond/AbstractSchwarz.hpp>
#include <random>

int test1(std::ofstream& fout)
{
    using namespace composyx;
    using Scalar = double;
    {
      const int N_DOFS = 8;
      const int N_SUBDOMAINS = 3;

      // Bind subdomains to MPI process
      std::shared_ptr<Process> p = bind_subdomains(N_SUBDOMAINS);

      // Define topology for each subdomain
      std::vector<Subdomain> sd;
      if(p->owns_subdomain(0)){
        std::map<int, std::vector<int>> nei_map_0 {{1, {1, 3, 4, 5}},
                                                   {2, {2, 3, 6, 7}}};
        std::map<int, std::vector<int>> nei_owner_map {{1, {0, 0, 1, 1}},
                                                   {2, {0, 0, 2, 2}}};
        sd.emplace_back(0, N_DOFS, std::move(nei_map_0),nei_owner_map,false);
      }
      if(p->owns_subdomain(1)){
        std::map<int, std::vector<int>> nei_map_1 {{0, {4, 5, 0, 2}},
                                                   {3, {2, 3, 6, 7}}};
        std::map<int, std::vector<int>> nei_owner_map {{0, {0, 0, 1, 1}},
                                                   {2, {1, 1, 2, 2}}};
        sd.emplace_back(1, N_DOFS, std::move(nei_map_1),nei_owner_map,false);
      }
      if(p->owns_subdomain(2)){
        std::map<int, std::vector<int>> nei_map_2 {{0, {4, 5, 0, 1}},
                                                   {3, {1, 3, 6, 7}}};
        std::map<int, std::vector<int>> nei_owner_map {{0, {0, 0, 2, 2}},
                                                   {3, {2, 2, 3, 3}}};
        sd.emplace_back(2, N_DOFS, std::move(nei_map_2),nei_owner_map,false);
      }
      if(p->owns_subdomain(3)){
        std::map<int, std::vector<int>> nei_map_3 {{1, {4, 5, 0, 1}},
                                                   {2, {6, 7, 0, 2}}};
        std::map<int, std::vector<int>> nei_owner_map {{1, {1, 1, 3, 3}},
                                                   {2, {2, 2, 3, 3}}};
        sd.emplace_back(3, N_DOFS, std::move(nei_map_3),nei_owner_map,false);
      }
      p->load_subdomains(sd);
      p->display("Process",fout) ;
    }
    return 0;
}

int test2(std::ofstream& fout)
{
    using namespace composyx;
    using Scalar = double;
    {
      /*
      0 0 11
      0 1 -0.5
      0 2 -0.5
      1 0 -0.5
      1 1 11
      1 3 -0.5
      2 0 -0.5
      2 2 11
      2 3 -0.5
      3 1 -0.5
      3 2 -0.5
      3 3 11
      */
        using Mat = SparseMatrixCOO<double> ;
        using Vec = Vector<double> ;
        SparseMatrixCOO<double> A({0,    0,   0,   1,  1,   1,   2,  2,   2,   3,   3,  3},
                                  {0,    1,   2,   0,  1,   3,   0,  2,   3,   1,   2,  3},
                                  {11,-0.5,-0.5,-0.5, 11,-0.5,-0.5, 11,-0.5,-0.5,-0.5, 11},
                                  4, 4);
        //A.set_spd(MatrixStorage::lower);

        Vector<double> u{1, 2, 3, 4};
        Vector<double> b = A * u;

        //ConjugateGradient<SparseMatrixCOO<double>, Vector<double>> solver;
        //solver.setup(parameters::A{A},
        //             parameters::verbose{true},
        //             parameters::max_iter{20},
        //             parameters::tolerance{1e-8});

        GMRES<Mat, Vec> solver;
        solver.setup( parameters::tolerance{1e-8},
                      parameters::max_iter{1000},
                      parameters::restart{2},
                      parameters::orthogonalization<Ortho>{Ortho::CGS},
                      parameters::verbose<bool>{true});
        solver.setup(A) ;

        auto x = solver * b;

        x.display("x");

      //SZ_compressor_init();
      //using DirectSolver = Mumps<SparseMatrixCSC<Scalar>, Vector<Scalar>>;
      //using DirectSolver = Pastix<SparseMatrixCOO<Scalar>, Vector<Scalar>> ;
      //pastix.set_n_threads(n_threads);
      //pastix.setup(A);
      //using PCD = AdditiveSchwarz<decltype(A), decltype(b), DirectSolver>;
      //GMRES<PMat, PVect> gmres(A);

    }
    return 0 ;
}

int main(int argc, char *argv[])
{
    using namespace composyx;
    using Scalar = double;

    MMPI::init();
    const int rank = MMPI::rank();
    std::stringstream filename;
    filename<<"output"<<rank<<".txt";
    std::ofstream fout(filename.str()) ;

    //test1(fout) ;
    test2(fout) ;

    MMPI::finalize();

    return 0;
}
