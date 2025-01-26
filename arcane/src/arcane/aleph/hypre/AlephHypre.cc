// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephHypre.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Implémentation Hypre de Aleph.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define HAVE_MPI
#define MPI_COMM_SUB (*(MPI_Comm*)(m_kernel->subParallelMng(m_index)->getMPICommunicator()))
#define OMPI_SKIP_MPICXX
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#endif
#include <HYPRE.h>
#include <HYPRE_utilities.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <_hypre_parcsr_mv.h>
#include <krylov.h>

#ifndef ItacRegion
#define ItacRegion(a, x)
#endif

#include "arcane/aleph/AlephArcane.h"

// Le type HYPRE_BigInt n'existe qu'à partir de Hypre 2.16.0
#if HYPRE_RELEASE_NUMBER < 21600
using HYPRE_BigInt = HYPRE_Int;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * NOTE: A partir de la version 2.14 de hypre (peut-être un peu avant),
 * hypre_TAlloc() et hypre_CAlloc() prennent un 3ème argument qui est
 * sur quel peripherique on alloue la mémoire (GPU ou CPU). Il n'y a pas
 * de moyens simples de savoir quelle est la version de hypre à partir
 * des .h mais par contre HYPRE_MEMORY_DEVICE et HYPRE_MEMORY_HOST sont
 * des macros donc on peut tester leur existance pour savoir s'il faut
 * appeler les méthodes hypre_TAlloc() et hypre_CAlloc() avec 2 ou 3 arguments.
 */
namespace
{
inline void
check(const char* hypre_func, HYPRE_Int error_code)
{
  if (error_code == 0)
    return;
  char buf[8192];
  HYPRE_DescribeError(error_code, buf);
  cout << "\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
       << "\nHYPRE ERROR in function "
       << hypre_func
       << "\nError_code=" << error_code
       << "\nMessage=" << buf
       << "\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
       << "\n"
       << std::flush;
  throw Exception("HYPRE Check", hypre_func);
}

template <typename T>
inline T*
_allocHypre(Integer size)
{
  size_t s = sizeof(T) * size;
  return reinterpret_cast<T*>(hypre_TAlloc(char, s, HYPRE_MEMORY_HOST));
}

template <typename T>
inline T*
_callocHypre(Integer size)
{
  size_t s = sizeof(T) * size;
  return reinterpret_cast<T*>(hypre_CTAlloc(char, s, HYPRE_MEMORY_HOST));
}

inline void
hypreCheck(const char* hypre_func, HYPRE_Int error_code)
{
  check(hypre_func, error_code);
  HYPRE_Int r = HYPRE_GetError();
  if (r != 0)
    cout << "HYPRE GET ERROR r=" << r
         << " error_code=" << error_code << " func=" << hypre_func << '\n';
}

} // namespace

/******************************************************************************
 AlephVectorHypre
 *****************************************************************************/
class AlephVectorHypre
: public IAlephVector
{
 public:
  AlephVectorHypre(ITraceMng* tm, AlephKernel* kernel, Integer index)
  : IAlephVector(tm, kernel, index)
  , jSize(0)
  , jUpper(0)
  , jLower(-1)
  {
    debug() << "[AlephVectorHypre::AlephVectorHypre] new SolverVectorHypre";
  }
  ~AlephVectorHypre()
  {
    if (m_hypre_ijvector)
      HYPRE_IJVectorDestroy(m_hypre_ijvector);
  }

 public:

  /******************************************************************************
   * The Create() routine creates an empty vector object that lives on the comm communicator. This is
   * a collective call, with each process passing its own index extents, jLower and jupper. The names
   * of these extent parameters begin with a j because we typically think of matrix-vector multiplies
   * as the fundamental operation involving both matrices and vectors. For matrix-vector multiplies,
   * the vector partitioning should match the column partitioning of the matrix (which also uses the j
   * notation). For linear system solves, these extents will typically match the row partitioning of the
   * matrix as well.
   *****************************************************************************/
  void AlephVectorCreate(void)
  {
    debug() << "[AlephVectorHypre::AlephVectorCreate] HYPRE VectorCreate";
    void* object;

    for (int iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
      if (m_kernel->rank() != m_kernel->solverRanks(m_index)[iCpu])
        continue;
      debug() << "[AlephVectorHypre::AlephVectorCreate] adding contibution of core #" << iCpu;
      if (jLower == -1)
        jLower = m_kernel->topology()->gathered_nb_row(iCpu);
      jUpper = m_kernel->topology()->gathered_nb_row(iCpu + 1) - 1;
    }

    debug() << "[AlephVectorHypre::AlephVectorCreate] jLower=" << jLower << ", jupper=" << jUpper;
    // Mise à jour de la taille locale du buffer pour le calcul plus tard de la norme max, par exemple
    jSize = jUpper - jLower + 1;

    hypreCheck("IJVectorCreate",
               HYPRE_IJVectorCreate(MPI_COMM_SUB,
                                    jLower,
                                    jUpper,
                                    &m_hypre_ijvector));

    debug() << "[AlephVectorHypre::AlephVectorCreate] HYPRE IJVectorSetObjectType";
    hypreCheck("IJVectorSetObjectType", HYPRE_IJVectorSetObjectType(m_hypre_ijvector, HYPRE_PARCSR));

    debug() << "[AlephVectorHypre::AlephVectorCreate] HYPRE IJVectorInitialize";
    hypreCheck("HYPRE_IJVectorInitialize", HYPRE_IJVectorInitialize(m_hypre_ijvector));

    HYPRE_IJVectorGetObject(m_hypre_ijvector, &object);
    m_hypre_parvector = (HYPRE_ParVector)object;
    debug() << "[AlephVectorHypre::AlephVectorCreate] done";
  }

  /******************************************************************************
   *****************************************************************************/
  void AlephVectorSet(const double* bfr_val, const AlephInt* bfr_idx, Integer size)
  {
    debug() << "[AlephVectorHypre::AlephVectorSet] size=" << size;
    hypreCheck("IJVectorSetValues", HYPRE_IJVectorSetValues(m_hypre_ijvector, size, bfr_idx, bfr_val));
  }

  /******************************************************************************
   *****************************************************************************/
  int AlephVectorAssemble(void)
  {
    debug() << "[AlephVectorHypre::AlephVectorAssemble]";
    hypreCheck("IJVectorAssemble", HYPRE_IJVectorAssemble(m_hypre_ijvector));
    return 0;
  }

  /******************************************************************************
   *****************************************************************************/
  void AlephVectorGet(double* bfr_val, const AlephInt* bfr_idx, Integer size)
  {
    //HYPRE_Int* hypre_bfr_idx = static
    debug() << "[AlephVectorHypre::AlephVectorGet] size=" << size;
    hypreCheck("HYPRE_IJVectorGetValues", HYPRE_IJVectorGetValues(m_hypre_ijvector, size, bfr_idx, bfr_val));
  }

  /******************************************************************************
  * norm_max
  *****************************************************************************/
  Real norm_max()
  {
    Real normInf = 0.0;
    UniqueArray<HYPRE_BigInt> bfr_idx(jSize);
    UniqueArray<double> bfr_val(jSize);

    for (HYPRE_Int i = 0; i < jSize; ++i)
      bfr_idx[i] = jLower + i;

    hypreCheck("HYPRE_IJVectorGetValues", HYPRE_IJVectorGetValues(m_hypre_ijvector, jSize, bfr_idx.data(), bfr_val.data()));
    for (HYPRE_Int i = 0; i < jSize; ++i) {
      const Real abs_val = math::abs(bfr_val[i]);
      if (abs_val > normInf)
        normInf = abs_val;
    }
    normInf = m_kernel->subParallelMng(m_index)->reduce(Parallel::ReduceMax, normInf);
    return normInf;
  }

  /******************************************************************************
   *****************************************************************************/
  void writeToFile(const String filename)
  {
    String filename_idx = filename; // + "_" + (int)m_kernel->subDomain()->commonVariables().globalIteration();
    debug() << "[AlephVectorHypre::writeToFile]";
    hypreCheck("HYPRE_IJVectorPrint",
               HYPRE_IJVectorPrint(m_hypre_ijvector, filename_idx.localstr()));
  }

 public:
  HYPRE_IJVector m_hypre_ijvector = nullptr;
  HYPRE_ParVector m_hypre_parvector = nullptr;
  HYPRE_Int jSize;
  HYPRE_Int jUpper;
  HYPRE_Int jLower;
};

/******************************************************************************
 AlephMatrixHypre
*****************************************************************************/
class AlephMatrixHypre
: public IAlephMatrix
{
 public:
  /******************************************************************************
 AlephMatrixHypre
  *****************************************************************************/
  AlephMatrixHypre(ITraceMng* tm, AlephKernel* kernel, Integer index)
  : IAlephMatrix(tm, kernel, index)
  , m_hypre_ijmatrix(0)
  {
    debug() << "[AlephMatrixHypre] new AlephMatrixHypre";
  }

  ~AlephMatrixHypre()
  {
    debug() << "[~AlephMatrixHypre]";
    if (m_hypre_ijmatrix)
      HYPRE_IJMatrixDestroy(m_hypre_ijmatrix);
  }

 public:
  /******************************************************************************
   * Each submatrix Ap is "owned" by a single process and its first and last row numbers are
   * given by the global indices ilower and iupper in the Create() call below.
   *******************************************************************************
   * The Create() routine creates an empty matrix object that lives on the comm communicator. This
   * is a collective call (i.e., must be called on all processes from a common synchronization point),
   * with each process passing its own row extents, ilower and iupper. The row partitioning must be
   * contiguous, i.e., iupper for process i must equal ilower-1 for process i+1. Note that this allows
   * matrices to have 0- or 1-based indexing. The parameters jlower and jupper define a column
   * partitioning, and should match ilower and iupper when solving square linear systems.
   *****************************************************************************/
  void AlephMatrixCreate(void)
  {
    debug() << "[AlephMatrixHypre::AlephMatrixCreate] HYPRE MatrixCreate idx:" << m_index;
    void* object;
    AlephInt ilower = -1;
    AlephInt iupper = 0;
    for (int iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
      if (m_kernel->rank() != m_kernel->solverRanks(m_index)[iCpu])
        continue;
      if (ilower == -1)
        ilower = m_kernel->topology()->gathered_nb_row(iCpu);
      iupper = m_kernel->topology()->gathered_nb_row(iCpu + 1) - 1;
    }
    debug() << "[AlephMatrixHypre::AlephMatrixCreate] ilower=" << ilower << ", iupper=" << iupper;

    AlephInt jlower = ilower; //0;
    AlephInt jupper = iupper; //m_kernel->topology()->gathered_nb_row(m_kernel->size())-1;
    debug() << "[AlephMatrixHypre::AlephMatrixCreate] jlower=" << jlower << ", jupper=" << jupper;

    hypreCheck("HYPRE_IJMatrixCreate",
               HYPRE_IJMatrixCreate(MPI_COMM_SUB,
                                    ilower, iupper,
                                    jlower, jupper,
                                    &m_hypre_ijmatrix));

    debug() << "[AlephMatrixHypre::AlephMatrixCreate] HYPRE IJMatrixSetObjectType";
    HYPRE_IJMatrixSetObjectType(m_hypre_ijmatrix, HYPRE_PARCSR);
    debug() << "[AlephMatrixHypre::AlephMatrixCreate] HYPRE IJMatrixSetRowSizes";
    HYPRE_IJMatrixSetRowSizes(m_hypre_ijmatrix, m_kernel->topology()->gathered_nb_row_elements().data());
    debug() << "[AlephMatrixHypre::AlephMatrixCreate] HYPRE IJMatrixInitialize";
    HYPRE_IJMatrixInitialize(m_hypre_ijmatrix);
    HYPRE_IJMatrixGetObject(m_hypre_ijmatrix, &object);
    m_hypre_parmatrix = (HYPRE_ParCSRMatrix)object;
  }

  /******************************************************************************
   *****************************************************************************/
  void AlephMatrixSetFilled(bool) {}

  /******************************************************************************
   *****************************************************************************/
  int AlephMatrixAssemble(void)
  {
    debug() << "[AlephMatrixHypre::AlephMatrixAssemble]";
    hypreCheck("HYPRE_IJMatrixAssemble",
               HYPRE_IJMatrixAssemble(m_hypre_ijmatrix));
    return 0;
  }

  /******************************************************************************
   *****************************************************************************/
  void AlephMatrixFill(int size, HYPRE_Int* rows, HYPRE_Int* cols, double* values)
  {
    debug() << "[AlephMatrixHypre::AlephMatrixFill] size=" << size;
    HYPRE_Int rtn = 0;
    HYPRE_Int col[1] = { 1 };
    for (int i = 0; i < size; i++) {
      rtn += HYPRE_IJMatrixSetValues(m_hypre_ijmatrix, 1, col, &rows[i], &cols[i], &values[i]);
    }
    hypreCheck("HYPRE_IJMatrixSetValues", rtn);
    //HYPRE_IJMatrixSetValues(m_hypre_ijmatrix, nrows, ncols, rows, cols, values);
    debug() << "[AlephMatrixHypre::AlephMatrixFill] done";
  }

  /******************************************************************************
   * isAlreadySolved
   *****************************************************************************/
  bool isAlreadySolved(AlephVectorHypre* x,
                       AlephVectorHypre* b,
                       AlephVectorHypre* tmp,
                       Real* residual_norm,
                       AlephParams* params)
  {
    HYPRE_ClearAllErrors();
    const bool convergence_analyse = params->convergenceAnalyse();

    // test le second membre du système linéaire
    const Real res0 = b->norm_max();

    if (convergence_analyse)
      info() << "analyse convergence : norme max du second membre res0 : " << res0;

    const Real considered_as_null = params->minRHSNorm();
    if (res0 < considered_as_null) {
      HYPRE_ParVectorSetConstantValues(x->m_hypre_parvector, 0.0);
      residual_norm[0] = res0;
      if (convergence_analyse)
        info() << "analyse convergence : le second membre du système linéaire est inférieur à : " << considered_as_null;
      return true;
    }

    if (params->xoUser()) {
      // on test si b est déjà solution du système à epsilon près
      //matrix->vectorProduct(b, tmp_vector); tmp_vector->sub(x);
      //M->vector_multiply(*tmp,*x);  // tmp=A*x
      //tmp->substract(*tmp,*b);      // tmp=A*x-b

      // X= alpha* M.B + beta * X (lu dans les sources de HYPRE)
      HYPRE_ParCSRMatrixMatvec(1.0, m_hypre_parmatrix, x->m_hypre_parvector, 0., tmp->m_hypre_parvector);
      HYPRE_ParVectorAxpy(-1.0, b->m_hypre_parvector, tmp->m_hypre_parvector);

      const Real residu = tmp->norm_max();
      //info() << "[IAlephHypre::isAlreadySolved] residu="<<residu;

      if (residu < considered_as_null) {
        if (convergence_analyse) {
          info() << "analyse convergence : |Ax0-b| est inférieur à " << considered_as_null;
          info() << "analyse convergence : x0 est déjà solution du système.";
        }
        residual_norm[0] = residu;
        return true;
      }

      const Real relative_error = residu / res0;
      if (convergence_analyse)
        info() << "analyse convergence : résidu initial : " << residu
               << " --- residu relatif initial (residu/res0) : " << residu / res0;

      if (relative_error < (params->epsilon())) {
        if (convergence_analyse)
          info() << "analyse convergence : X est déjà solution du système";
        residual_norm[0] = residu;
        return true;
      }
    }
    return false;
  }

  /******************************************************************************
   *****************************************************************************/
  int AlephMatrixSolve(AlephVector* x,
                       AlephVector* b,
                       AlephVector* t,
                       Integer& nb_iteration,
                       Real* residual_norm,
                       AlephParams* solver_param)
  {
    solver_param->setAmgCoarseningMethod(TypesSolver::AMG_COARSENING_AUTO);
    const String func_name("SolverMatrixHypre::solve");
    void* object;
    int ierr = 0;

    auto* ximpl = ARCANE_CHECK_POINTER(dynamic_cast<AlephVectorHypre*>(x->implementation()));
    auto* bimpl = ARCANE_CHECK_POINTER(dynamic_cast<AlephVectorHypre*>(b->implementation()));

    HYPRE_IJVector solution = ximpl->m_hypre_ijvector;
    HYPRE_IJVector RHS = bimpl->m_hypre_ijvector;
    //HYPRE_IJVector tmp      = (dynamic_cast<AlephVectorHypre*> (t->implementation()))->m_hypre_ijvector;

    HYPRE_IJMatrixGetObject(m_hypre_ijmatrix, &object);
    HYPRE_ParCSRMatrix M = (HYPRE_ParCSRMatrix)object;
    HYPRE_IJVectorGetObject(solution, &object);
    HYPRE_ParVector X = (HYPRE_ParVector)object;
    HYPRE_IJVectorGetObject(RHS, &object);
    HYPRE_ParVector B = (HYPRE_ParVector)object;
    //HYPRE_IJVectorGetObject(tmp,&object);
    //HYPRE_ParVector T = (HYPRE_ParVector)object;

    auto* ximpl2 = ARCANE_CHECK_POINTER(dynamic_cast<AlephVectorHypre*>(x->implementation()));
    auto* bimpl2 = ARCANE_CHECK_POINTER(dynamic_cast<AlephVectorHypre*>(b->implementation()));
    auto* timpl2 = ARCANE_CHECK_POINTER(dynamic_cast<AlephVectorHypre*>(t->implementation()));
    if (isAlreadySolved(ximpl2, bimpl2, timpl2, residual_norm, solver_param)) {
      ItacRegion(isAlreadySolved, AlephMatrixHypre);
      debug() << "[AlephMatrixHypre::AlephMatrixSolve] isAlreadySolved !";
      nb_iteration = 0;
      return 0;
    }

    TypesSolver::ePreconditionerMethod preconditioner_method = solver_param->precond();
    //  TypesSolver::ePreconditionerMethod preconditioner_method = TypesSolver::NONE;
    TypesSolver::eSolverMethod solver_method = solver_param->method();

    // déclaration et initialisation du solveur
    HYPRE_Solver solver = 0;

    switch (solver_method) {
    case TypesSolver::PCG:
      initSolverPCG(solver_param, solver);
      break;
    case TypesSolver::BiCGStab:
      initSolverBiCGStab(solver_param, solver);
      break;
    case TypesSolver::GMRES:
      initSolverGMRES(solver_param, solver);
      break;
    default:
      throw ArgumentException(func_name, "solveur inconnu");
    }

    // déclaration et initialisation du preconditionneur
    HYPRE_Solver precond = 0;

    switch (preconditioner_method) {
    case TypesSolver::NONE:
      break;
    case TypesSolver::DIAGONAL:
      setDiagonalPreconditioner(solver_method, solver, precond);
      break;
    case TypesSolver::ILU:
      setILUPreconditioner(solver_method, solver, precond);
      break;
    case TypesSolver::SPAIstat:
      setSpaiStatPreconditioner(solver_method, solver, solver_param, precond);
      break;
    case TypesSolver::AMG:
      setAMGPreconditioner(solver_method, solver, solver_param, precond);
      break;
    case TypesSolver::AINV:
      throw ArgumentException(func_name, "preconditionnement AINV indisponible");
    case TypesSolver::SPAIdyn:
      throw ArgumentException(func_name, "preconditionnement SPAIdyn indisponible");
    case TypesSolver::ILUp:
      throw ArgumentException(func_name, "preconditionnement ILUp indisponible");
    case TypesSolver::IC:
      throw ArgumentException(func_name, "preconditionnement IC indisponible");
    case TypesSolver::POLY:
      throw ArgumentException(func_name, "preconditionnement POLY indisponible");
    default:
      throw ArgumentException(func_name, "preconditionnement inconnu");
    }

    // résolution du système algébrique
    HYPRE_Int iteration = 0;
    double residue = 0.0;

    switch (solver_method) {
    case TypesSolver::PCG:
      ierr = solvePCG(solver_param, solver, M, B, X, iteration, residue);
      break;
    case TypesSolver::BiCGStab:
      ierr = solveBiCGStab(solver, M, B, X, iteration, residue);
      break;
    case TypesSolver::GMRES:
      ierr = solveGMRES(solver, M, B, X, iteration, residue);
      break;
    default:
      ierr = -3;
      return ierr;
    }
    nb_iteration = static_cast<Integer>(iteration);
    residual_norm[0] = static_cast<Real>(residue);

    /* for(int i=0;i<8;++i){
   int idx[1];
   double valx[1]={-1.};
   //double valb[1]={-1.};
   idx[0]=i;
   HYPRE_IJVectorGetValues(solution, 1, idx, valx);
   debug()<<"[AlephMatrixHypre::AlephMatrixSolve] X["<<i<<"]="<<valx[0];
   //(static_cast<AlephVectorHypre*> (x->implementation()))->AlephVectorGet(valx,idx,1);
   //debug()<<"[AlephMatrixHypre::AlephMatrixSolve] x["<<i<<"]="<<valx[0];
   //HYPRE_IJVectorGetValues(RHS, 1, idx, valb);
   //debug()<<"[AlephMatrixHypre::AlephMatrixSolve] B["<<i<<"]="<<valb[0];
   }
*/
    switch (preconditioner_method) {
    case TypesSolver::NONE:
      break;
    case TypesSolver::DIAGONAL:
      break;
    case TypesSolver::ILU:
      HYPRE_ParCSRPilutDestroy(precond);
      break;
    case TypesSolver::SPAIstat:
      HYPRE_ParCSRParaSailsDestroy(precond);
      break;
    case TypesSolver::AMG:
      HYPRE_BoomerAMGDestroy(precond);
      break;
    default:
      throw ArgumentException(func_name, "preconditionnement inconnu");
    }

    if (iteration == solver_param->maxIter() && solver_param->stopErrorStrategy()) {
      info() << "\n============================================================";
      info() << "\nCette erreur est retournée après " << iteration << "\n";
      info() << "\nOn a atteind le nombre max d'itérations du solveur.";
      info() << "\nIl est possible de demander au code de ne pas tenir compte de cette erreur.";
      info() << "\nVoir la documentation du jeu de données concernant le service solveur.";
      info() << "\n======================================================";
      throw Exception("AlephMatrixHypre::Solve", "On a atteind le nombre max d'itérations du solveur");
    }
    return ierr;
  }

  /******************************************************************************
 *****************************************************************************/
  void writeToFile(const String filename)
  {
    HYPRE_IJMatrixPrint(m_hypre_ijmatrix, filename.localstr());
  }

  /******************************************************************************
 *****************************************************************************/
  void initSolverPCG(const AlephParams* solver_param, HYPRE_Solver& solver)
  {
    const String func_name = "SolverMatrixHypre::initSolverPCG";
    double epsilon = solver_param->epsilon();
    int max_it = solver_param->maxIter();
    int output_level = solver_param->getOutputLevel();

    HYPRE_ParCSRPCGCreate(MPI_COMM_SUB, &solver);
    HYPRE_ParCSRPCGSetMaxIter(solver, max_it);
    HYPRE_ParCSRPCGSetTol(solver, epsilon);
    HYPRE_ParCSRPCGSetTwoNorm(solver, 1);
    HYPRE_ParCSRPCGSetPrintLevel(solver, output_level);
  }

  /******************************************************************************
   *****************************************************************************/
  void initSolverBiCGStab(const AlephParams* solver_param, HYPRE_Solver& solver)
  {
    const String func_name = "SolverMatrixHypre::initSolverBiCGStab";
    double epsilon = solver_param->epsilon();
    int max_it = solver_param->maxIter();
    int output_level = solver_param->getOutputLevel();

    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_SUB, &solver);
    HYPRE_ParCSRBiCGSTABSetMaxIter(solver, max_it);
    HYPRE_ParCSRBiCGSTABSetTol(solver, epsilon);
    HYPRE_ParCSRBiCGSTABSetPrintLevel(solver, output_level);
  }

  /******************************************************************************
   *****************************************************************************/
  void initSolverGMRES(const AlephParams* solver_param, HYPRE_Solver& solver)
  {
    const String func_name = "SolverMatrixHypre::initSolverGMRES";
    double epsilon = solver_param->epsilon();
    int max_it = solver_param->maxIter();
    int output_level = solver_param->getOutputLevel();

    HYPRE_ParCSRGMRESCreate(MPI_COMM_SUB, &solver);
    const int krylov_dim = 20; // dimension Krylov space for GMRES
    HYPRE_ParCSRGMRESSetKDim(solver, krylov_dim);
    HYPRE_ParCSRGMRESSetMaxIter(solver, max_it);
    HYPRE_ParCSRGMRESSetTol(solver, epsilon);
    HYPRE_ParCSRGMRESSetPrintLevel(solver, output_level);
  }

  /******************************************************************************
   *****************************************************************************/
  void setDiagonalPreconditioner(const TypesSolver::eSolverMethod solver_method,
                                 const HYPRE_Solver& solver,
                                 HYPRE_Solver& precond)
  {
    const String func_name = "SolverMatrixHypre::setDiagonalPreconditioner";
    switch (solver_method) {
    case TypesSolver::PCG:
      HYPRE_ParCSRPCGSetPrecond(solver,
                                HYPRE_ParCSRDiagScale,
                                HYPRE_ParCSRDiagScaleSetup,
                                precond);
      break;
    case TypesSolver::BiCGStab:
      HYPRE_ParCSRBiCGSTABSetPrecond(solver,
                                     HYPRE_ParCSRDiagScale,
                                     HYPRE_ParCSRDiagScaleSetup,
                                     precond);
      break;
    case TypesSolver::GMRES:
      HYPRE_ParCSRGMRESSetPrecond(solver,
                                  HYPRE_ParCSRDiagScale,
                                  HYPRE_ParCSRDiagScaleSetup,
                                  precond);
      break;
    default:
      throw ArgumentException(func_name, "solveur inconnu pour preconditionnement 'Diagonal'");
    }
  }

  /******************************************************************************
   *****************************************************************************/
  void setILUPreconditioner(const TypesSolver::eSolverMethod solver_method,
                            const HYPRE_Solver& solver,
                            HYPRE_Solver& precond)
  {
    const String func_name = "SolverMatrixHypre::setILUPreconditioner";
    switch (solver_method) {
    case TypesSolver::PCG:
      throw ArgumentException(func_name, "solveur PCG indisponible avec le preconditionnement 'ILU'");
      break;
    case TypesSolver::BiCGStab:
      HYPRE_ParCSRPilutCreate(MPI_COMM_SUB, &precond);
      HYPRE_ParCSRBiCGSTABSetPrecond(solver,
                                     HYPRE_ParCSRPilutSolve,
                                     HYPRE_ParCSRPilutSetup,
                                     precond);
      break;
    case TypesSolver::GMRES:
      HYPRE_ParCSRPilutCreate(MPI_COMM_SUB,
                              &precond);
      HYPRE_ParCSRGMRESSetPrecond(solver,
                                  HYPRE_ParCSRPilutSolve,
                                  HYPRE_ParCSRPilutSetup,
                                  precond);
      break;
    default:
      throw ArgumentException(func_name, "solveur inconnu pour preconditionnement ILU\n");
    }
  }

  /******************************************************************************
   *****************************************************************************/
  void setSpaiStatPreconditioner(const TypesSolver::eSolverMethod solver_method,
                                 const HYPRE_Solver& solver,
                                 const AlephParams* solver_param,
                                 HYPRE_Solver& precond)
  {
    HYPRE_ParCSRParaSailsCreate(MPI_COMM_SUB, &precond);
    double alpha = solver_param->alpha();
    int gamma = solver_param->gamma();
    if (alpha < 0.0)
      alpha = 0.1; // valeur par defaut pour le parametre de tolerance
    if (gamma == -1)
      gamma = 1; // valeur par defaut pour le parametre de remplissage
    HYPRE_ParCSRParaSailsSetParams(precond, alpha, gamma);
    switch (solver_method) {
    case TypesSolver::PCG:
      HYPRE_ParCSRPCGSetPrecond(solver, HYPRE_ParCSRParaSailsSolve, HYPRE_ParCSRParaSailsSetup, precond);
      break;
    case TypesSolver::BiCGStab:
      throw ArgumentException("AlephMatrixHypre::setSpaiStatPreconditioner", "solveur 'BiCGStab' invalide pour preconditionnement 'SPAIstat'");
      break;
    case TypesSolver::GMRES:
      // matrice non symétrique
      HYPRE_ParCSRParaSailsSetSym(precond, 0);
      HYPRE_ParCSRGMRESSetPrecond(solver, HYPRE_ParaSailsSolve, HYPRE_ParaSailsSetup, precond);
      break;
    default:
      throw ArgumentException("AlephMatrixHypre::setSpaiStatPreconditioner", "solveur inconnu pour preconditionnement 'SPAIstat'\n");
      break;
    }
  }

  /******************************************************************************
   *****************************************************************************/
  void setAMGPreconditioner(const TypesSolver::eSolverMethod solver_method,
                            const HYPRE_Solver& solver,
                            const AlephParams* solver_param,
                            HYPRE_Solver& precond)
  {
    // defaults for BoomerAMG from hypre example -- lc
    // TODO : options and defaults values must be completed
    double trunc_factor = 0.1; // set AMG interpolation truncation factor = val
    int cycle_type = solver_param->getAmgCycle(); // set AMG cycles (1=V, 2=W, etc.)
    int coarsen_type = solver_param->amgCoarseningMethod();
    // Ruge coarsening (local) if <val> == 1
    int relax_default = 3; // relaxation type <val> :
    //        0=Weighted Jacobi
    //        1=Gauss-Seidel (very slow!)
    //        3=Hybrid Jacobi/Gauss-Seidel
    int num_sweep = 1; // Use <val> sweeps on each level (here 1)
    int hybrid = 1; // no switch in coarsening if -1
    int measure_type = 1; // use globale measures
    double max_row_sum = 1.0; // set AMG maximum row sum threshold for dependency weakening

    int max_levels = 50; // 25;  // maximum number of AMG levels
    const int gamma = solver_param->gamma();
    if (gamma != -1)
      max_levels = gamma; // utilisation de la valeur du jeu de donnees

    double strong_threshold = 0.1; // 0.25; // set AMG threshold Theta = val
    const double alpha = solver_param->alpha();
    if (alpha > 0.0)
      strong_threshold = alpha; // utilisation de la valeur du jeu de donnees
    // news
    Integer output_level = solver_param->getOutputLevel();

    HYPRE_Int* num_grid_sweeps = _allocHypre<HYPRE_Int>(4);
    HYPRE_Int* grid_relax_type = _allocHypre<HYPRE_Int>(4);
    HYPRE_Int** grid_relax_points = _allocHypre<HYPRE_Int*>(4);
    double* relax_weight = _allocHypre<double>(max_levels);

    for (int i = 0; i < max_levels; i++)
      relax_weight[i] = 1.0;

    if (coarsen_type == 5) {
      /* fine grid */
      num_grid_sweeps[0] = 3;
      grid_relax_type[0] = relax_default;
      grid_relax_points[0] = _allocHypre<HYPRE_Int>(3);
      grid_relax_points[0][0] = -2;
      grid_relax_points[0][1] = -1;
      grid_relax_points[0][2] = 1;

      /* down cycle */
      num_grid_sweeps[1] = 4;
      grid_relax_type[1] = relax_default;
      grid_relax_points[1] = _callocHypre<HYPRE_Int>(4);
      grid_relax_points[1][0] = -1;
      grid_relax_points[1][1] = 1;
      grid_relax_points[1][2] = -2;
      grid_relax_points[1][3] = -2;

      /* up cycle */
      num_grid_sweeps[2] = 4;
      grid_relax_type[2] = relax_default;
      grid_relax_points[2] = _allocHypre<HYPRE_Int>(4);
      grid_relax_points[2][0] = -2;
      grid_relax_points[2][1] = -2;
      grid_relax_points[2][2] = 1;
      grid_relax_points[2][3] = -1;
    }
    else {
      /* fine grid */
      num_grid_sweeps[0] = 2 * num_sweep;
      grid_relax_type[0] = relax_default;
      grid_relax_points[0] = _allocHypre<HYPRE_Int>(2 * num_sweep);
      for (int i = 0; i < 2 * num_sweep; i += 2) {
        grid_relax_points[0][i] = -1;
        grid_relax_points[0][i + 1] = 1;
      }

      /* down cycle */
      num_grid_sweeps[1] = 2 * num_sweep;
      grid_relax_type[1] = relax_default;
      grid_relax_points[1] = _allocHypre<HYPRE_Int>(2 * num_sweep);
      for (int i = 0; i < 2 * num_sweep; i += 2) {
        grid_relax_points[1][i] = -1;
        grid_relax_points[1][i + 1] = 1;
      }

      /* up cycle */
      num_grid_sweeps[2] = 2 * num_sweep;
      grid_relax_type[2] = relax_default;
      grid_relax_points[2] = _allocHypre<HYPRE_Int>(2 * num_sweep);
      for (int i = 0; i < 2 * num_sweep; i += 2) {
        grid_relax_points[2][i] = -1;
        grid_relax_points[2][i + 1] = 1;
      }
    }

    /* coarsest grid */
    num_grid_sweeps[3] = 1;
    grid_relax_type[3] = 9;
    grid_relax_points[3] = _allocHypre<HYPRE_Int>(1);
    grid_relax_points[3][0] = 0;

    // end of default seting

    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetPrintLevel(precond, output_level);
    HYPRE_BoomerAMGSetCoarsenType(precond, (hybrid * coarsen_type));
    HYPRE_BoomerAMGSetMeasureType(precond, measure_type);
    HYPRE_BoomerAMGSetStrongThreshold(precond, strong_threshold);
    HYPRE_BoomerAMGSetTruncFactor(precond, trunc_factor);
    HYPRE_BoomerAMGSetMaxIter(precond, 1);
    HYPRE_BoomerAMGSetCycleType(precond, cycle_type);
    HYPRE_BoomerAMGSetNumGridSweeps(precond, num_grid_sweeps);
    HYPRE_BoomerAMGSetGridRelaxType(precond, grid_relax_type);
    HYPRE_BoomerAMGSetRelaxWeight(precond, relax_weight);
    HYPRE_BoomerAMGSetGridRelaxPoints(precond, grid_relax_points);
    HYPRE_BoomerAMGSetTol(precond, 0.0);
    HYPRE_BoomerAMGSetMaxLevels(precond, max_levels);
    HYPRE_BoomerAMGSetMaxRowSum(precond, max_row_sum);

    switch (solver_method) {
    case TypesSolver::PCG:
      HYPRE_ParCSRPCGSetPrecond(solver,
                                HYPRE_BoomerAMGSolve,
                                HYPRE_BoomerAMGSetup,
                                precond);
      break;
    case TypesSolver::BiCGStab:
      HYPRE_ParCSRBiCGSTABSetPrecond(solver,
                                     HYPRE_BoomerAMGSolve,
                                     HYPRE_BoomerAMGSetup,
                                     precond);
      break;
    case TypesSolver::GMRES:
      HYPRE_ParCSRGMRESSetPrecond(solver,
                                  HYPRE_BoomerAMGSolve,
                                  HYPRE_BoomerAMGSetup,
                                  precond);
      break;
    default:
      throw ArgumentException("AlephMatrixHypre::setAMGPreconditioner", "solveur inconnu pour preconditionnement 'AMG'\n");
    }
  }

  /******************************************************************************
   *****************************************************************************/
  bool solvePCG(const AlephParams* solver_param,
                HYPRE_Solver& solver,
                HYPRE_ParCSRMatrix& M,
                HYPRE_ParVector& B,
                HYPRE_ParVector& X,
                HYPRE_Int& iteration,
                double& residue)
  {
    const String func_name = "SolverMatrixHypre::solvePCG";
    const bool xo = solver_param->xoUser();
    bool error = false;

    if (!xo)
      HYPRE_ParVectorSetConstantValues(X, 0.0);
    HYPRE_ParCSRPCGSetup(solver, M, B, X);
    HYPRE_ParCSRPCGSolve(solver, M, B, X);
    HYPRE_ParCSRPCGGetNumIterations(solver, &iteration);
    HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(solver, &residue);

    HYPRE_Int converged = 0;
    HYPRE_PCGGetConverged(solver, &converged);
    error |= (!converged);

    HYPRE_ParCSRPCGDestroy(solver);

    return !error;
  }

  /******************************************************************************
   *****************************************************************************/
  bool solveBiCGStab(HYPRE_Solver& solver,
                     HYPRE_ParCSRMatrix& M,
                     HYPRE_ParVector& B,
                     HYPRE_ParVector& X,
                     HYPRE_Int& iteration,
                     double& residue)
  {
    const String func_name = "SolverMatrixHypre::solveBiCGStab";
    bool error = false;
    HYPRE_ParVectorSetRandomValues(X, 775);
    HYPRE_ParCSRBiCGSTABSetup(solver, M, B, X);
    HYPRE_ParCSRBiCGSTABSolve(solver, M, B, X);
    HYPRE_ParCSRBiCGSTABGetNumIterations(solver, &iteration);
    HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver, &residue);

    HYPRE_Int converged = 0;
    hypre_BiCGSTABGetConverged(solver, &converged);
    error |= (!converged);

    HYPRE_ParCSRBiCGSTABDestroy(solver);

    return !error;
  }

  /******************************************************************************
   *****************************************************************************/
  bool solveGMRES(HYPRE_Solver& solver,
                  HYPRE_ParCSRMatrix& M,
                  HYPRE_ParVector& B,
                  HYPRE_ParVector& X,
                  HYPRE_Int& iteration,
                  double& residue)
  {
    const String func_name = "SolverMatrixHypre::solveGMRES";
    bool error = false;
    HYPRE_ParCSRGMRESSetup(solver, M, B, X);
    HYPRE_ParCSRGMRESSolve(solver, M, B, X);
    HYPRE_ParCSRGMRESGetNumIterations(solver, &iteration);
    HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver, &residue);

    HYPRE_Int converged = 0;
    HYPRE_GMRESGetConverged(solver, &converged);
    error |= (!converged);

    HYPRE_ParCSRGMRESDestroy(solver);
    return !error;
  }

 private:

  HYPRE_IJMatrix m_hypre_ijmatrix = nullptr;
  HYPRE_ParCSRMatrix m_hypre_parmatrix = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HypreAlephFactoryImpl
: public AbstractService
, public IAlephFactoryImpl
{
 public:
  HypreAlephFactoryImpl(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}
  ~HypreAlephFactoryImpl()
  {
    for ( auto* v : m_IAlephVectors )
      delete v;
    for ( auto* v : m_IAlephMatrixs )
      delete v;
  }

 public:

  void initialize() override
  {
    // NOTE: A partir de la 2.29, on peut utiliser
    // HYPRE_Initialize() et tester si l'initialisation
    // a déjà été faite via HYPRE_Initialized().
#if HYPRE_RELEASE_NUMBER >= 22900
    if (!HYPRE_Initialized()){
      info() << "Initializing HYPRE";
      HYPRE_Initialize();
    }
#elif HYPRE_RELEASE_NUMBER >= 22700
    info() << "Initializing HYPRE";
    HYPRE_Init();
#endif

#if HYPRE_RELEASE_NUMBER >= 22700
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
#endif
  }

  IAlephTopology* createTopology(ITraceMng* tm,
                                 AlephKernel* kernel,
                                 Integer index,
                                 Integer nb_row_size) override
  {
    ARCANE_UNUSED(tm);
    ARCANE_UNUSED(kernel);
    ARCANE_UNUSED(index);
    ARCANE_UNUSED(nb_row_size);
    return NULL;
  }

  IAlephVector* createVector(ITraceMng* tm,
                             AlephKernel* kernel,
                             Integer index) override
  {
    IAlephVector* new_vector = new AlephVectorHypre(tm, kernel, index);
    m_IAlephVectors.add(new_vector);
    return new_vector;
  }

  IAlephMatrix* createMatrix(ITraceMng* tm,
                             AlephKernel* kernel,
                             Integer index) override
  {
    IAlephMatrix* new_matrix = new AlephMatrixHypre(tm, kernel, index);
    m_IAlephMatrixs.add(new_matrix);
    return new_matrix;
  }

 private:
  UniqueArray<IAlephVector*> m_IAlephVectors;
  UniqueArray<IAlephMatrix*> m_IAlephMatrixs;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(HypreAlephFactoryImpl,IAlephFactoryImpl,HypreAlephFactory);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
