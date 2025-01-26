// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * IAlephTrilinos.h                                            (C) 2010-2023 *
 * constants for output types
 #define AZ_all             -4  Print out everything including matrix
 #define AZ_none             0  Print out no results (not even warnings)
 #define AZ_last            -1  Print out final residual and warnings
 #define AZ_summary         -2  Print out summary, final residual and warnings
 #define AZ_warnings        -3  Print out only warning messages
 *****************************************************************************/

#include "arcane/aleph/AlephArcane.h"

#include "Epetra_config.h"
#include "Epetra_Vector.h"
#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "AztecOO.h"
#include "ml_MultiLevelPreconditioner.h"
#include "Ifpack_IC.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephVectorTrilinos : public IAlephVector
{
 public:

  AlephVectorTrilinos(ITraceMng* tm, AlephKernel* kernel, Integer index)
  : IAlephVector(tm, kernel, index)
  {
    debug() << "\t\t[AlephVectorTrilinos::AlephVectorTrilinos] new SolverVectorTrilinos";
#ifdef HAVE_MPI
    m_trilinos_Comm = new Epetra_MpiComm(*(MPI_Comm*)(m_kernel->subParallelMng(m_index)->getMPICommunicator()));
#else
    m_trilinos_Comm = new Epetra_SerialComm();
#endif // HAVE_MPI
  }

  ~AlephVectorTrilinos()
  {
    delete m_trilinos_vector;
    delete m_trilinos_Comm;
  }

 public:

  /******************************************************************************
 * AlephVectorCreate
 *****************************************************************************/
  void AlephVectorCreate(void)
  {
    debug() << "\t\t[AlephVectorTrilinos::AlephVectorCreate] TRILINOS VectorCreate";
    Integer jlower = -1;
    Integer jupper = 0;
    for (int iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
      if (m_kernel->rank() != m_kernel->solverRanks(m_index)[iCpu])
        continue;
      if (jlower == -1)
        jlower = m_kernel->topology()->gathered_nb_row(iCpu);
      jupper = m_kernel->topology()->gathered_nb_row(iCpu + 1) - 1;
    }
    Integer size = jupper - jlower + 1;
    debug() << "\t\t[AlephVectorTrilinos::AlephVectorCreate] jlower=" << jlower << ", jupper=" << jupper;
    Epetra_Map trilinos_Map = Epetra_Map(m_kernel->topology()->nb_row_size(),
                                         size,
                                         0,
                                         *m_trilinos_Comm);
    m_trilinos_vector = new Epetra_Vector(trilinos_Map);
    debug() << "\t\t[AlephVectorTrilinos::AlephVectorCreate] done";
  }

  /******************************************************************************
 * AlephVectorSet
 *****************************************************************************/
  void AlephVectorSet(const double* bfr_val, const int* bfr_idx, Integer size)
  {
    debug() << "\t\t[AlephVectorTrilinos::AlephVectorSet]";
    m_trilinos_vector->ReplaceGlobalValues(size, bfr_val, bfr_idx);
    /*#warning DUMPING trilinos
    for(int i=0; i<size;++i){
      debug()<<"\t\t[AlephVectorTrilinos::AlephVectorSet] @["<<bfr_idx[i]<<"]="<<bfr_val[i];
      }*/
  }

  int AlephVectorAssemble(void)
  {
    //m_trilinos_vector->FillComplete();
    return 0;
  }

  /******************************************************************************
 * AlephVectorGet
 *****************************************************************************/
  void AlephVectorGet(double* bfr_val, const int* bfr_idx, Integer size)
  {
    ARCANE_UNUSED(bfr_idx);
    debug() << "\t\t[AlephVectorTrilinos::AlephVectorGet]";
    for (int i = 0; i < size; ++i) {
      //	 bfr_val[i]=(*m_trilinos_vector)[bfr_idx[i]];
      bfr_val[i] = (*m_trilinos_vector)[i];
      /*#warning DUMPING trilinos
  debug()<<"\t\t[AlephVectorTrilinos::AlephVectorGet] @["<<i<<"]="<<bfr_val[i];
*/
    }
  }

  /******************************************************************************
 * writeToFile
 *****************************************************************************/
  void writeToFile(const String filename)
  {
    ARCANE_UNUSED(filename);
    debug() << "\t\t[AlephVectorTrilinos::writeToFile]";
  }

  /******************************************************************************
 * LinftyNorm
 * int NormInf (double * Result) const;
 *****************************************************************************/
  Real LinftyNorm(void)
  {
    Real Result;
    if (m_trilinos_vector->NormInf(&Result) != 0)
      throw Exception("LinftyNorm", "NormInf error");
    return Result;
  }

  /******************************************************************************
 * LinftyNorm
 *****************************************************************************/
  void fill(Real value)
  {
    m_trilinos_vector->PutScalar(value);
  }

 public:
  Epetra_Vector* m_trilinos_vector = nullptr;
  Epetra_Comm* m_trilinos_Comm = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephMatrixTrilinos
: public IAlephMatrix
{
 public:
  /******************************************************************************
 AlephMatrixTrilinos
  *****************************************************************************/
  AlephMatrixTrilinos(ITraceMng* tm, AlephKernel* kernel, Integer index)
  : IAlephMatrix(tm, kernel, index)
  {
    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixTrilinos] new AlephMatrixTrilinos";
#ifdef HAVE_MPI
    m_trilinos_Comm = new Epetra_MpiComm(*(MPI_Comm*)(m_kernel->subParallelMng(m_index)->getMPICommunicator()));
#else
    m_trilinos_Comm = new Epetra_SerialComm();
#endif // HAVE_MPI
  }

  ~AlephMatrixTrilinos()
  {
    delete m_trilinos_matrix;
    delete m_trilinos_Comm;
  }

  /******************************************************************************
 * AlephMatrixCreate
 *****************************************************************************/
  void AlephMatrixCreate(void)
  {
    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixCreate] TRILINOS MatrixCreate idx:" << m_index;
    Integer ilower = -1;
    Integer iupper = 0;
    for ( int iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
      if (m_kernel->rank() != m_kernel->solverRanks(m_index)[iCpu])
        continue;
      if (ilower == -1)
        ilower = m_kernel->topology()->gathered_nb_row(iCpu);
      iupper = m_kernel->topology()->gathered_nb_row(iCpu + 1) - 1;
    }
    Integer size = iupper - ilower + 1;

    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixCreate] ilower=" << ilower << ", iupper=" << iupper;
    Integer jlower = ilower;
    Integer jupper = iupper;

    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixCreate] jlower=" << jlower << ", jupper=" << jupper;
    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixCreate] global=" << m_kernel->topology()->nb_row_size();
    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixCreate] size=" << size;

    Epetra_Map trilinos_Map = Epetra_Map(m_kernel->topology()->nb_row_size(),
                                         size,
                                         0,
                                         *m_trilinos_Comm);
    m_trilinos_matrix = new Epetra_CrsMatrix(Copy,
                                             trilinos_Map,
                                             m_kernel->topology()->gathered_nb_row_elements().subView(ilower, size).unguardedBasePointer(),
                                             false);
  }

  /******************************************************************************
 * AlephMatrixSetFilled
 *****************************************************************************/
  void AlephMatrixSetFilled(bool) {}

  /******************************************************************************
 * AlephMatrixConfigure
 *****************************************************************************/
  int AlephMatrixAssemble(void)
  {
    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixAssemble] AlephMatrixAssemble";
    m_trilinos_matrix->FillComplete();
    return true;
  }

  /******************************************************************************
 * AlephMatrixFill
 *****************************************************************************/
  void AlephMatrixFill(int size, int* rows, int* cols, double* values)
  {
    [[maybe_unused]] int rtn = 0;
    for (int i = 0; i < size; i++) {
      // int InsertGlobalValues(int GlobalRow, int NumEntries, double* Values, int* Indices);
      /*#warning TRILINOS DUMP
      debug()<<"\t\t[AlephMatrixTrilinos::AlephMatrixFill] A["<<rows[i]<<"]["<<cols[i]<<"]="<<values[i];
*/
      rtn += m_trilinos_matrix->InsertGlobalValues(rows[i], 1, &values[i], &cols[i]);
    }

    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixFill] done";
  }

  /******************************************************************************
   * LinftyNormVectorProductAndSub
   *****************************************************************************/
  Real LinftyNormVectorProductAndSub(AlephVector* x,
                                     AlephVector* b)
  {
    ARCANE_UNUSED(x);
    ARCANE_UNUSED(b);
    throw FatalErrorException("LinftyNormVectorProductAndSub", "error");
  }

  /******************************************************************************
 * isAlreadySolved
 *****************************************************************************/
  bool isAlreadySolved(AlephVectorTrilinos* x,
                       AlephVectorTrilinos* b,
                       AlephVectorTrilinos* tmp,
                       Real* residual_norm,
                       AlephParams* params)
  {
    const bool convergence_analyse = true; //params->convergenceAnalyse();

    // test le second membre du système linéaire
    const Real res0 = b->LinftyNorm();

    if (convergence_analyse)
      debug() << "analyse convergence : norme max du second membre res0 : " << res0;

    const Real considered_as_null = params->minRHSNorm();
    if (res0 < considered_as_null) {
      x->fill(Real(0.0));
      residual_norm[0] = res0;
      if (convergence_analyse)
        debug() << "analyse convergence : le second membre du système linéaire est inférieur à : " << considered_as_null;
      return true;
    }

    if (params->xoUser()) {
      // on test si b est déjà solution du système à epsilon près
      //matrix->vectorProduct(b, tmp_vector); tmp_vector->sub(x);
      m_trilinos_matrix->Multiply(false,
                                  *x->m_trilinos_vector,
                                  *tmp->m_trilinos_vector); // tmp=A*x
      tmp->m_trilinos_vector->Update(-1.0,
                                     *b->m_trilinos_vector,
                                     1.0); // tmp=A*x-b
      const Real residu = tmp->LinftyNorm();
      //debug() << "[IAlephTrilinos::isAlreadySolved] residu="<<residu;

      if (residu < considered_as_null) {
        if (convergence_analyse) {
          debug() << "analyse convergence : |Ax0-b| est inférieur à " << considered_as_null;
          debug() << "analyse convergence : x0 est déjà solution du système.";
        }
        residual_norm[0] = residu;
        return true;
      }

      const Real relative_error = residu / res0;
      if (convergence_analyse)
        debug() << "analyse convergence : résidu initial : " << residu
                << " --- residu relatif initial (residu/res0) : " << residu / res0;

      if (relative_error < (params->epsilon())) {
        if (convergence_analyse)
          debug() << "analyse convergence : X est déjà solution du système";
        residual_norm[0] = residu;
        return true;
      }
    }
    if (convergence_analyse)
      debug() << "analyse convergence : return false";
    return false;
  }

  /******************************************************************************
 * AlephMatrixSolve
 *****************************************************************************/
  int AlephMatrixSolve(AlephVector* x,
                       AlephVector* b,
                       AlephVector* t,
                       Integer& nb_iteration,
                       Real* residual_norm,
                       AlephParams* solver_param)
  {
    Ifpack_IC* ICPrecond = nullptr;
    Teuchos::ParameterList* MLList = nullptr;
    ML_Epetra::MultiLevelPreconditioner* MLPrecond = nullptr;
    const String func_name("SolverMatrixTrilinos::solve");

    AlephVectorTrilinos* x_tri = dynamic_cast<AlephVectorTrilinos*>(x->implementation());
    AlephVectorTrilinos* b_tri = dynamic_cast<AlephVectorTrilinos*>(b->implementation());
    AlephVectorTrilinos* t_tri = dynamic_cast<AlephVectorTrilinos*>(t->implementation());

    ARCANE_CHECK_POINTER(x_tri);
    ARCANE_CHECK_POINTER(b_tri);
    ARCANE_CHECK_POINTER(t_tri);

    if (isAlreadySolved(x_tri, b_tri, t_tri, residual_norm, solver_param)) {
      ItacRegion(isAlreadySolved, AlephMatrixSloop);
      debug() << "\t[AlephMatrixSloop::AlephMatrixSolve] isAlreadySolved !";
      nb_iteration = 0;
      return 0;
    }

    Epetra_Vector* X = x_tri->m_trilinos_vector;
    Epetra_Vector* B = b_tri->m_trilinos_vector;

    if (!solver_param->xoUser())
      x_tri->fill(0.0);

    // Create Linear Problem
    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixSolve] Create Linear Problem";
    Epetra_LinearProblem problem(m_trilinos_matrix, X, B);

    // Create AztecOO instance
    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixSolve] Create AztecOO instance";
    AztecOO solver(problem);
    // Options can be: AZ_none AZ_last AZ_summary AZ_warnings
    solver.SetAztecOption(AZ_output, solver_param->getOutputLevel());

    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixSolve] Setting options";
    switch (solver_param->method()) {
    case TypesSolver::PCG:
      solver.SetAztecOption(AZ_solver, AZ_cg);
      break;
    case TypesSolver::BiCGStab:
      solver.SetAztecOption(AZ_solver, AZ_bicgstab);
      break;
    case TypesSolver::BiCGStab2:
      solver.SetAztecOption(AZ_solver, AZ_bicgstab);
      break;
    case TypesSolver::GMRES:
      solver.SetAztecOption(AZ_solver, AZ_gmres);
      break;
    case TypesSolver::SuperLU:
      solver.SetAztecOption(AZ_solver, AZ_slu);
      break;
    default:
      throw ArgumentException(func_name, "solveur inconnu");
    }

    switch (solver_param->precond()) {
    case TypesSolver::NONE:
      solver.SetAztecOption(AZ_precond, AZ_none);
      break;
    case TypesSolver::DIAGONAL:
      solver.SetAztecOption(AZ_precond, AZ_Jacobi);
      break;
    case TypesSolver::ILU: {
      solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
      solver.SetAztecOption(AZ_subdomain_solve, AZ_ilu);
      break;
    }
    case TypesSolver::ILUp:
      solver.SetAztecOption(AZ_precond, AZ_bilu);
      break;
    case TypesSolver::POLY:
      solver.SetAztecOption(AZ_precond, AZ_Neumann);
      break;
    case TypesSolver::AMG: { // Taken from trilinos-x.y.z-Source/packages/ml/examples/BasicExamples/*
      MLList = new Teuchos::ParameterList();
      /* Setting parameter for aggregation-based preconditioners:
         - "SA" : classical smoothed aggregation preconditioners;
         - "NSSA" : default values for Petrov-Galerkin preconditioner for nonsymmetric systems
         - "maxwell" : default values for aggregation preconditioner for eddy current systems
         - "DD" : defaults for 2-level domain decomposition preconditioners based on aggregation;
         - "DD-LU" : Like "DD", but use exact LU decompositions on each subdomain;
         - "DD-ML" : 3-level domain decomposition preconditioners, with coarser spaces defined by aggregation;
         - "DD-ML-LU" : Like "DD-ML", but with LU decompositions on each subdomain.
      */
      ML_Epetra::SetDefaults("SA", *MLList);
      //MLList->set("ML output", 10);       // output level, 0 being silent and 10 verbose
      //MLList->set("max levels",16);       // maximum number of levels
      //MLList->set("increasing or decreasing","increasing");                // set finest level to 0
      MLList->set("cycle applications", solver_param->getAmgSolverIter());
      debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixSolve] Setting cycle application=" << solver_param->getAmgSolverIter();
      //MLList->set("aggregation: type", "MIS");                             // use MIS scheme to create the aggregate
      //MLList->set("smoother: type","Chebyshev");                           // smoother is Chebyshev
      MLList->set("smoother: sweeps", solver_param->getAmgSmootherIter());
      debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixSolve] Setting smoother: sweeps=" << solver_param->getAmgSmootherIter();
      //MLList->set("smoother: pre or post", "both");                        // use both pre and post smoothing
      //MLList->set("coarse: type","Amesos-KLU");                            // solve with serial direct solver KLU
      //MLList->set("coarse: max size",32);                                  // maximum allowed coarse size
      MLList->set("ML debug mode", false);

      MLPrecond = new ML_Epetra::MultiLevelPreconditioner(*m_trilinos_matrix, *MLList);
      //MLPrecond->PrintUnused(0); // verify unused parameters on process 0 (put -1 to print on all processes)
      //MLPrecond->AnalyzeHierarchy(true,1,1,1);
      // MLPrec->ReComputePreconditioner();                   // Cheap recompute the preconditioner
      // It is supposed that the linear system matrix has different values, but
      // **exactly** the same structure and layout. The code re-built the
      // hierarchy and re-setup the smoothers and the coarse solver using
      // already available information on the hierarchy. A particular
      // care is required to use ReComputePreconditioner() with nonzero threshold.
      // tell AztecOO to use the ML preconditioner
      solver.SetPrecOperator(MLPrecond);
      break;
    }
    case TypesSolver::IC: {
      ICPrecond = new Ifpack_IC(m_trilinos_matrix);
      IFPACK_CHK_ERR(ICPrecond->Compute());
      solver.SetPrecOperator(ICPrecond);
      break;
    }
    case TypesSolver::SPAIstat:
      throw ArgumentException(func_name, "preconditionnement AztecOO::SPAIstat indisponible");
    case TypesSolver::AINV:
      throw ArgumentException(func_name, "preconditionnement AztecOO::AINV indisponible");
    case TypesSolver::SPAIdyn:
      throw ArgumentException(func_name, "preconditionnement AztecOO::SPAIdyn indisponible");
    default:
      throw ArgumentException(func_name, "preconditionnement inconnu");
    }

    // Déclenchement du solver
    // Iterates on the current problem until MaxIters or Tolerance is reached.
    solver.Iterate(solver_param->maxIter(), solver_param->epsilon());

    double norm[1];
    solver.GetLHS()->Norm2(norm);
    double real_residual;
    X->Norm2(&real_residual);

    debug() << "[AlephMatrixTrilinos::AlephMatrixSolve]"
            // Returns the total number of iterations performed on this problem
            << "\n\t\tNumIters=" << solver.NumIters()
            // Returns the true unscaled residual for this problem
            << "\n\t\tTrueResidual=" << solver.TrueResidual()
            // Returns the true scaled residual for this problem
            << "\n\t\tScaledResidual=" << solver.ScaledResidual()
            // Returns the recursive residual for this problem
            << "\n\t\tRecursiveResidual=" << solver.RecursiveResidual()
            << "\n\t\tnorm=" << norm[0]
            << "\n\t\tRealResidual=" << real_residual;

    nb_iteration = static_cast<Integer>(solver.NumIters());
    residual_norm[0] = static_cast<Real>(solver.TrueResidual()); // vs ScaledResidual ?

    if (solver_param->maxIter() <= nb_iteration)
      throw Exception("Nombre max d'itérations du solveur atteint!",
                      "AlephMatrixTrilinos::AlephMatrixSolve");
    /*if (solver_param->epsilon()<residual_norm[0])
      throw Exception("Convergence non atteinte!", "AlephMatrixTrilinos::AlephMatrixSolve");
    */

    if (MLList != NULL)
      delete MLList;
    if (MLPrecond != NULL)
      delete MLPrecond;
    if (ICPrecond != NULL)
      delete ICPrecond;

    debug() << "\t\t[AlephMatrixTrilinos::AlephMatrixSolve] done";
    return nb_iteration;
  }

  /******************************************************************************
 * writeToFile
 *****************************************************************************/
  void writeToFile(const String filename)
  {
    ARCANE_UNUSED(filename);
  }

 private:
  Epetra_CrsMatrix* m_trilinos_matrix = nullptr;
  Epetra_Comm* m_trilinos_Comm = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TrilinosAlephFactoryImpl : public AbstractService
, public IAlephFactoryImpl
{
 public:
  TrilinosAlephFactoryImpl(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  , m_IAlephVectors(0)
  , m_IAlephMatrixs(0)
  {}
  ~TrilinosAlephFactoryImpl()
  {
    for ( auto* v : m_IAlephVectors )
      delete v;
    for ( auto* v : m_IAlephMatrixs )
      delete v;
  }

 public:
  virtual void initialize() {}
  virtual IAlephTopology* createTopology(ITraceMng* tm,
                                         AlephKernel* kernel,
                                         Integer index,
                                         Integer nb_row_size)
  {
    ARCANE_UNUSED(tm);
    ARCANE_UNUSED(kernel);
    ARCANE_UNUSED(index);
    ARCANE_UNUSED(nb_row_size);
    return nullptr;
  }
  virtual IAlephVector* createVector(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     Integer index)
  {
    IAlephVector* new_vector = new AlephVectorTrilinos(tm, kernel, index);
    m_IAlephVectors.add(new_vector);
    return new_vector;
  }

  virtual IAlephMatrix* createMatrix(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     Integer index)
  {
    IAlephMatrix* new_matrix = new AlephMatrixTrilinos(tm, kernel, index);
    m_IAlephMatrixs.add(new_matrix);
    return new_matrix;
  }

 private:
  UniqueArray<IAlephVector*> m_IAlephVectors;
  UniqueArray<IAlephMatrix*> m_IAlephMatrixs;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(TrilinosAlephFactoryImpl,IAlephFactoryImpl,TrilinosAlephFactory);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
