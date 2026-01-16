// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephPETSc.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Implémentation PETSc de Aleph.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/AlephArcane.h"

#include "arcane/utils/StringList.h"

#define MPI_COMM_SUB *(MPI_Comm*)(m_kernel->subParallelMng(m_index)->getMPICommunicator())

#include "petscksp.h"
#include "petscsys.h"

// TODO: mieux gérer les sous-versions de PETSC
#if PETSC_VERSION_GE(3,6,1)
#include "petscmat.h"
#elif PETSC_VERSION_(3,3,0)
#include "petscpcmg.h"
#elif PETSC_VERSION_(3,0,0)
#include "petscmg.h"
#else
#error "Bad version of PETSc: should be 3.0.0, 3.3.0 or greater than 3.6.1."
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
// To convert from AlephInt data to PetscInt data
PetscInt* _toPetscInt(AlephInt* v)
{
  static_assert(sizeof(PetscInt)==sizeof(AlephInt),"AlephInt and PetscInt should have the same size");
  return reinterpret_cast<PetscInt*>(v);
}
// To convert from AlephInt data to PetscInt data
const PetscInt* _toPetscInt(const AlephInt* v)
{
  static_assert(sizeof(PetscInt)==sizeof(AlephInt),"AlephInt and PetscInt should have the same size");
  return reinterpret_cast<const PetscInt*>(v);
}

inline void
petscCheck(PetscErrorCode error_code)
{
  // Note GG: Je ne sais pas vraiment à partir de quelle version de PETSc
  // la valeur 'PETSC_SUCCESS' la fonction 'PetscCallVoid' sont
  // disponibles mais elles le sont dans la 3.18.0.
#if PETSC_VERSION_GE(3,19,0)
  if (error_code==PETSC_SUCCESS)
    return;
  PetscCallVoid(error_code);
  ARCANE_FATAL("Error in PETSc backend errcode={0}",error_code);
#else
  if (error_code!=0)
    ARCANE_FATAL("Error in PETSc backend errcode={0}",error_code);
#endif
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephTopologyPETSc
: public IAlephTopology
{
 public:

  AlephTopologyPETSc(ITraceMng* tm, AlephKernel* kernel, Integer index, Integer nb_row_size);
  ~AlephTopologyPETSc() override
  {
    debug() << "\33[1;5;32m\t\t\t[~AlephTopologyPETSc]\33[0m";
  }

 public:

  void backupAndInitialize() override {}
  void restore() override {}

 private:

  void _checkInitPETSc();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephVectorPETSc
: public IAlephVector
{
 public:

  AlephVectorPETSc(ITraceMng*, AlephKernel*, Integer);
  ~AlephVectorPETSc();
  void AlephVectorCreate(void) override;
  void AlephVectorSet(const double*, const AlephInt*, Integer) override;
  int AlephVectorAssemble(void) override;
  void AlephVectorGet(double*, const AlephInt*, Integer) override;
  void writeToFile(const String) override;
  Real LinftyNorm(void);

 public:

  Vec m_petsc_vector = nullptr;
  PetscInt jSize = 0;
  PetscInt jUpper = 0;
  PetscInt jLower = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephMatrixPETSc
: public IAlephMatrix
{
 public:

  AlephMatrixPETSc(ITraceMng*, AlephKernel*, Integer);
  ~AlephMatrixPETSc();
  void AlephMatrixCreate(void) override;
  void AlephMatrixSetFilled(bool) override;
  int AlephMatrixAssemble(void) override;
  void AlephMatrixFill(int, AlephInt*, AlephInt*, double*) override;
  Real LinftyNormVectorProductAndSub(AlephVector*, AlephVector*);
  bool isAlreadySolved(AlephVectorPETSc*, AlephVectorPETSc*,
                       AlephVectorPETSc*, Real*, AlephParams*);
  int AlephMatrixSolve(AlephVector*, AlephVector*,
                       AlephVector*, Integer&, Real*, AlephParams*) override;
  void writeToFile(const String) override;

 private:

  Mat m_petsc_matrix = nullptr;
  KSP m_ksp_solver = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
bool global_need_petsc_finalize = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PETScAlephFactoryImpl
: public AbstractService
, public IAlephFactoryImpl
{
 public:
  explicit PETScAlephFactoryImpl(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}
  ~PETScAlephFactoryImpl() override
  {
    for (auto* v : m_IAlephVectors)
      delete v;
    for (auto* v : m_IAlephMatrixs)
      delete v;
    for (auto* v : m_IAlephTopologys)
      delete v;
    if (global_need_petsc_finalize){
      // TODO: Il faudrait appeler PETscFinalize() mais il
      // faut le faire faire avant le MPI_Finalize() ce qui
      // n'est pas toujours le cas suivant quand cette fabrique
      // est détruite. En attendant, on n'appelle pas la fonction
      //PetscFinalize();
      global_need_petsc_finalize = false;
    }
  }

 public:

  void initialize() override {}
  IAlephTopology* createTopology(ITraceMng* tm,
                                 AlephKernel* kernel,
                                 Integer index,
                                 Integer nb_row_size) override
  {
    IAlephTopology* new_topology = new AlephTopologyPETSc(tm, kernel, index, nb_row_size);
    m_IAlephTopologys.add(new_topology);
    return new_topology;
  }
  IAlephVector* createVector(ITraceMng* tm,
                             AlephKernel* kernel,
                             Integer index) override
  {
    IAlephVector* new_vector = new AlephVectorPETSc(tm, kernel, index);
    m_IAlephVectors.add(new_vector);
    return new_vector;
  }

  IAlephMatrix* createMatrix(ITraceMng* tm,
                             AlephKernel* kernel,
                             Integer index) override
  {
    IAlephMatrix* new_matrix = new AlephMatrixPETSc(tm, kernel, index);
    m_IAlephMatrixs.add(new_matrix);
    return new_matrix;
  }

 private:

  UniqueArray<IAlephVector*> m_IAlephVectors;
  UniqueArray<IAlephMatrix*> m_IAlephMatrixs;
  UniqueArray<IAlephTopology*> m_IAlephTopologys;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlephTopologyPETSc::
AlephTopologyPETSc(ITraceMng* tm, AlephKernel* kernel, Integer index, Integer nb_row_size)
: IAlephTopology(tm, kernel, index, nb_row_size)
{
  ARCANE_CHECK_POINTER(kernel);
  if (!m_participating_in_solver) {
    debug() << "\33[1;32m\t[AlephTopologyPETSc] Not concerned with this solver, returning\33[0m";
    return;
  }
  debug() << "\33[1;32m\t\t[AlephTopologyPETSc] @" << this << "\33[0m";
  if (!m_kernel->isParallel()) {
    PETSC_COMM_WORLD = PETSC_COMM_SELF;
  }
  else {
    PETSC_COMM_WORLD = *(MPI_Comm*)(kernel->subParallelMng(index)->getMPICommunicator());
  }

  _checkInitPETSc();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlephTopologyPETSc::
_checkInitPETSc()
{
  // Ne fait rien si PETSc a déjà été initialisé.
  PetscBool is_petsc_initialized = PETSC_FALSE;
  PetscInitialized(&is_petsc_initialized);
  if (is_petsc_initialized==PETSC_TRUE)
    return;

  AlephKernelSolverInitializeArguments& init_args = m_kernel->solverInitializeArgs();
  bool has_args = init_args.hasValues();
  debug() << Trace::Color::cyan() << "[AlephTopologyPETSc] Initializing PETSc. UseArg=" << has_args;

  if (has_args){
    const CommandLineArguments& args = init_args.commandLineArguments();
    PetscInitialize(args.commandLineArgc(),args.commandLineArgv(),nullptr,nullptr);
  }
  else{
    PetscInitializeNoArguments();
  }
  global_need_petsc_finalize = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// ****************************************************************************
// * AlephVectorPETSc
// ****************************************************************************
AlephVectorPETSc::
AlephVectorPETSc(ITraceMng *tm,AlephKernel *kernel,Integer index)
: IAlephVector(tm,kernel,index)
, m_petsc_vector()
, jSize(0)
, jUpper(0)
, jLower(-1)
{
  debug()<<"\t\t[AlephVectorPETSc::AlephVectorPETSc] new SolverVectorPETSc";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlephVectorPETSc::
~AlephVectorPETSc()
{
  VecDestroy(&m_petsc_vector);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// ****************************************************************************
// * AlephVectorCreate
// ****************************************************************************
void AlephVectorPETSc::
AlephVectorCreate(void)
{
  for( int iCpu=0;iCpu<m_kernel->size();++iCpu){
    if (m_kernel->rank()!=m_kernel->solverRanks(m_index)[iCpu])
      continue;
    if (jLower==-1)
      jLower=m_kernel->topology()->gathered_nb_row(iCpu);
    jUpper=m_kernel->topology()->gathered_nb_row(iCpu+1);
  }
  // Mise à jour de la taille locale du buffer pour le calcul plus tard de la norme max, par exemple
  jSize=jUpper-jLower;
  VecCreateMPI(MPI_COMM_SUB,
               jSize, // n
               m_kernel->topology()->nb_row_size(), // N
               &m_petsc_vector);
  debug()<<"\t\t[AlephVectorPETSc::AlephVectorCreate] PETSC VectorCreate"
        <<" of local size=" <<jSize<<"/"<<m_kernel->topology()->nb_row_size()
        <<", done";
}

// ****************************************************************************
// * AlephVectorSet
// ****************************************************************************
void AlephVectorPETSc::
AlephVectorSet(const double *bfr_val, const AlephInt *bfr_idx, Integer size)
{
  VecSetValues(m_petsc_vector,size,_toPetscInt(bfr_idx),bfr_val,INSERT_VALUES);
  debug()<<"\t\t[AlephVectorPETSc::AlephVectorSet] "<<size<<" values inserted!";
  // En séquentiel, il faut le fair aussi avec PETSc
  AlephVectorAssemble();
}


// ****************************************************************************
// * AlephVectorAssemble
// ****************************************************************************
int AlephVectorPETSc::
AlephVectorAssemble(void)
{
  VecAssemblyBegin(m_petsc_vector);
  VecAssemblyEnd(m_petsc_vector);
  debug()<<"\t\t[AlephVectorPETSc::AlephVectorAssemble]";
  return 0;
}


// ****************************************************************************
// * AlephVectorGet
// ****************************************************************************
void AlephVectorPETSc::
AlephVectorGet(double* bfr_val, const AlephInt* bfr_idx, Integer size)
{
  VecGetValues(m_petsc_vector,size,_toPetscInt(bfr_idx),bfr_val);
  debug()<<"\t\t[AlephVectorPETSc::AlephVectorGet] fetched "<< size <<" values!";
}


// ****************************************************************************
// * writeToFile
// ****************************************************************************
void AlephVectorPETSc::
writeToFile(const String filename)
{
  ARCANE_UNUSED(filename);
  debug()<<"\t\t[AlephVectorPETSc::writeToFile]";
}

// ****************************************************************************
// * LinftyNorm
// ****************************************************************************
Real AlephVectorPETSc::
LinftyNorm(void)
{
  PetscReal val;
  VecNorm(m_petsc_vector,NORM_INFINITY,&val);
  return val;
}

// ****************************************************************************
// * AlephMatrixPETSc
// ****************************************************************************

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlephMatrixPETSc::
AlephMatrixPETSc(ITraceMng *tm, AlephKernel *kernel, Integer index)
: IAlephMatrix(tm,kernel,index)
{
  debug()<<"\t\t[AlephMatrixPETSc::AlephMatrixPETSc] new AlephMatrixPETSc";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlephMatrixPETSc::
~AlephMatrixPETSc()
{
  if (m_petsc_matrix)
    MatDestroy(&m_petsc_matrix);
  if (m_ksp_solver)
    KSPDestroy(&m_ksp_solver);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// ****************************************************************************
// * AlephMatrixCreate
// ****************************************************************************
void AlephMatrixPETSc::
AlephMatrixCreate(void)
{
  AlephInt ilower=-1;
  AlephInt iupper=0;

  for( int iCpu=0;iCpu<m_kernel->size();++iCpu ){
    if (m_kernel->rank()!=m_kernel->solverRanks(m_index)[iCpu]) continue;
    if (ilower==-1) ilower=m_kernel->topology()->gathered_nb_row(iCpu);
    iupper=m_kernel->topology()->gathered_nb_row(iCpu+1);
  }

  AlephInt size = iupper-ilower;

  PetscInt* row_data = _toPetscInt(m_kernel->topology()->gathered_nb_row_elements().subView(ilower,size).data());

  PetscInt m = size;  // Local number of rows
  PetscInt n = size;  // Local number of columns (assuming square)
  PetscInt M = m_kernel->topology()->nb_row_size(); // Global number of rows
  PetscInt N = M; // Global number of columns (assuming square)

  MatCreate(PETSC_COMM_WORLD, &m_petsc_matrix);
  MatSetSizes(m_petsc_matrix, m, n, M, N);

  // Preallocate (AIJ is sparse, so preallocation improves performance)
  MatSetType(m_petsc_matrix, MATAIJ);
  MatMPIAIJSetPreallocation(m_petsc_matrix, 0, row_data, 0, row_data);

  // Allow matrix type selection at runtime (Flexible, Runtime Configurable with -mat_type)
  MatSetFromOptions(m_petsc_matrix);
#if PETSC_VERSION_LE(3,0,0)
  AlephInt jlower=ilower;
  AlephInt jupper=iupper;

  PETSC_VERSION_MatCreate(MPI_COMM_SUB,
                          iupper-ilower, // m = number of rows
                          jupper-jlower, // n = number of columns
                          m_kernel->topology()->nb_row_size(), // M = number of global rows
                          m_kernel->topology()->nb_row_size(), // N = number of global columns
                          0, // ignored (number of nonzeros per row in DIAGONAL portion of local submatrix)
                          // array containing the number of nonzeros in the various rows of the DIAGONAL portion of local submatrix
                          row_data,
                          0, // ignored (number of nonzeros per row in the OFF-DIAGONAL portion of local submatrix)
                          // array containing the number of nonzeros in the various rows of the OFF-DIAGONAL portion of local submatrix
                          row_data,
                          &m_petsc_matrix);
#endif

  MatSetOption(m_petsc_matrix, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
  MatSetUp(m_petsc_matrix);
  debug()<<"\t\t[AlephMatrixPetsc::AlephMatrixCreate] PETSC MatrixCreate idx:"<<m_index
         <<", ("<<ilower<<"->"<<(iupper-1)<<")";
}


// ****************************************************************************
// * AlephMatrixSetFilled
// ****************************************************************************
void AlephMatrixPETSc::
AlephMatrixSetFilled(bool)
{
  debug()<<"\t\t[AlephMatrixPETSc::AlephMatrixSetFilled] done";
  MatSetOption(m_petsc_matrix, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);
}


// ****************************************************************************
// * AlephMatrixAssemble
// ****************************************************************************
int AlephMatrixPETSc::
AlephMatrixAssemble(void)
{
  debug()<<"\t\t[AlephMatrixPETSc::AlephMatrixAssemble] AlephMatrixAssemble";
  MatAssemblyBegin(m_petsc_matrix,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(m_petsc_matrix,MAT_FINAL_ASSEMBLY);
  return 0;
}


// ****************************************************************************
// * AlephMatrixFill
// ****************************************************************************
void AlephMatrixPETSc::
AlephMatrixFill(int size, AlephInt* rows, AlephInt* cols, double *values)
{
  debug()<<"\t\t[AlephMatrixPETSc::AlephMatrixFill] size="<<size;
  for(int i=0;i<size;i++){
    //debug()<<"\t\t[AlephMatrixPETSc::AlephMatrixFill] i="<<i;
    MatSetValue(m_petsc_matrix, rows[i], cols[i], values[i], INSERT_VALUES);
  }
  debug()<<"\t\t[AlephMatrixPETSc::AlephMatrixFill] done";
  // PETSc réclame systématiquement l'assemblage
  AlephMatrixAssemble();
}



// ****************************************************************************
// * LinftyNormVectorProductAndSub
// ****************************************************************************
Real AlephMatrixPETSc::
LinftyNormVectorProductAndSub(AlephVector* x,AlephVector* b)
{
  ARCANE_UNUSED(x);
  ARCANE_UNUSED(b);
  throw FatalErrorException("LinftyNormVectorProductAndSub", "error");
}


// ****************************************************************************
// * isAlreadySolved
// ****************************************************************************
bool AlephMatrixPETSc::
isAlreadySolved(AlephVectorPETSc* x,
                AlephVectorPETSc* b,
                AlephVectorPETSc* tmp,
                Real* residual_norm,
                AlephParams* params)
{
  ARCANE_UNUSED(x);
  ARCANE_UNUSED(b);
  ARCANE_UNUSED(tmp);
  ARCANE_UNUSED(residual_norm);
  ARCANE_UNUSED(params);
  return false;
}


// ****************************************************************************
// * AlephMatrixSolve
// ****************************************************************************
int AlephMatrixPETSc::
AlephMatrixSolve(AlephVector* x,AlephVector* b,AlephVector* t,
                 Integer& nb_iteration,Real* residual_norm,
                 AlephParams* solver_param)
{
  ARCANE_UNUSED(t);

  PC prec;
  PetscInt its;
  PetscReal norm;
  KSPConvergedReason reason;

  AlephVectorPETSc* x_petsc = dynamic_cast<AlephVectorPETSc*> (x->implementation());
  AlephVectorPETSc* b_petsc = dynamic_cast<AlephVectorPETSc*> (b->implementation());

  ARCANE_CHECK_POINTER(x_petsc);
  ARCANE_CHECK_POINTER(b_petsc);
  const bool is_parallel = m_kernel->subParallelMng(m_index)->isParallel();

  Vec solution = x_petsc->m_petsc_vector;
  Vec RHS      = b_petsc->m_petsc_vector;

  debug()<<"[AlephMatrixSolve]";
  if (m_ksp_solver)
    KSPDestroy(&m_ksp_solver);
  KSPCreate(MPI_COMM_SUB,&m_ksp_solver);
#if PETSC_VERSION_GE(3,6,1)
  KSPSetOperators(m_ksp_solver,m_petsc_matrix,m_petsc_matrix);//SAME_PRECONDITIONER);
#else
  KSPSetOperators(m_ksp_solver,m_petsc_matrix,m_petsc_matrix,SAME_NONZERO_PATTERN);//SAME_PRECONDITIONER);
#endif
  // Builds KSP for a particular solver
  switch(solver_param->method()){
    // Preconditioned conjugate gradient (PCG) iterative method
  case TypesSolver::PCG      : KSPSetType(m_ksp_solver,KSPCG); break;
    // BiCGStab (Stabilized version of BiConjugate Gradient Squared) method
  case TypesSolver::BiCGStab : KSPSetType(m_ksp_solver,KSPBCGS); break;
    // IBiCGStab (Improved Stabilized version of BiConjugate Gradient Squared) method
    // in an alternative form to have only a single global reduction operation instead of the usual 3 (or 4)
  case TypesSolver::BiCGStab2: KSPSetType(m_ksp_solver,KSPIBCGS); break;
    // Generalized Minimal Residual method. (Saad and Schultz, 1986) with restart
  case TypesSolver::GMRES    : KSPSetType(m_ksp_solver,KSPGMRES); break;
  default : throw ArgumentException("AlephMatrixPETSc::AlephMatrixSolve", "Unknown solver method");
  }

  KSPGetPC(m_ksp_solver,&prec);

  switch(solver_param->precond()){
  case TypesSolver::NONE     : PCSetType(prec,PCNONE); break;
    // Jacobi (i.e. diagonal scaling preconditioning)
  case TypesSolver::DIAGONAL : PCSetType(prec,PCJACOBI); break;
    // Incomplete factorization preconditioners.
  case TypesSolver::ILU:
    // ILU ne fonctionne pas en parallèle avec PETSc (sauf si compilé avec Hypre)
    // ILU can be set up in parallel via a sub-preconditioner to Block-Jacobi preconditioner
    if (is_parallel)
      PCSetType(prec,PCBJACOBI);
    else
      PCSetType(prec,PCILU);
    break;
    // Incomplete Cholesky factorization preconditioners.
  case TypesSolver::IC:
    // IC can be set up in parallel via a sub-preconditioner to Block-Jacobi preconditioner
    if (is_parallel) {
      PCSetType(prec,PCBJACOBI);     // set PC to Block Jacobi
      PetscInt    nlocal, first;
      KSP        *subksp;            // KSP context for subdomain
      PC          subpc;             // PC context for subdomain
      KSPSetUp(m_ksp_solver);
      PCBJacobiGetSubKSP(prec, &nlocal, &first, &subksp); // Extract the KSP context array for the local blocks
      for (int i = 0; i < nlocal; i++) {
        KSPGetPC(subksp[i], &subpc);
        PCSetType(subpc, PCICC);         // set subdomain PC to IC
        }
      }
    else
      PCSetType(prec,PCICC);
    break;
    // Sparse Approximate Inverse method of Grote and Barnard as a preconditioner (SIAM J. Sci. Comput.; vol 18, nr 3)
  case TypesSolver::SPAIstat : PCSetType(prec,PCSPAI); break;
    // Use multigrid preconditioning.
    // This preconditioner requires you provide additional information
    // about the coarser grid matrices and restriction/interpolation operators.
  case TypesSolver::AMG:
    // By default PCMG uses GMRES on the fine grid smoother so this should be used with KSPFGMRES
    // or the smoother changed to not use GMRES
    // Implements the Flexible Generalized Minimal Residual method. developed by Saad with restart
    KSPSetType(m_ksp_solver,KSPFGMRES);
    PCSetType(prec,PCMG);
    //Sets the number of levels to use with MG. Must be called before any other MG routine.
    PCMGSetLevels(prec,1,
                  (MPI_Comm*)(m_kernel->subParallelMng(m_index)->getMPICommunicator()));
    // Determines the form of multigrid to use: multiplicative, additive, full, or the Kaskade algorithm.
    PCMGSetType(prec,PC_MG_MULTIPLICATIVE); // PC_MG_MULTIPLICATIVE,PC_MG_ADDITIVE,PC_MG_FULL,PC_MG_KASKADE
    // Sets the type cycles to use. Use PCMGSetCycleTypeOnLevel() for more complicated cycling.
    PCMGSetCycleType(prec,PC_MG_CYCLE_V);   // PC_MG_CYCLE_V or PC_MG_CYCLE_W
    //Sets the number of pre-smoothing steps to use on all levels
#if PETSC_VERSION_GE(3,10,2)
    PCMGSetNumberSmooth(prec, 1);
#else
    PCMGSetNumberSmoothDown(prec, 1);
    PCMGSetNumberSmoothUp(prec, 1);
#endif
    // PCMGSetResidual: Sets the function to be used to calculate the residual on the lth level
    // PCMGSetInterpolation: Sets the function to be used to calculate the interpolation from l-1 to the lth level
    // PCMGSetRestriction: Sets the function to be used to restrict vector from level l to l-1
    // PCMGSetRhs: Sets the vector space to be used to store the right-hand side on a particular level
    // PCMGSetX: Sets the vector space to be used to store the solution on a particular level
    // PCMGSetR: Sets the vector space to be used to store the residual on a particular level
    //PCMGSetRestriction, PCMGSetRhs, PCMGSetX, PCMGSetR, etc.
    //PCMGSetLevels
    break;
  case TypesSolver::AINV    : throw ArgumentException("AlephMatrixPETSc", "preconditionnement AINV indisponible");
  case TypesSolver::SPAIdyn : throw ArgumentException("AlephMatrixPETSc", "preconditionnement SPAIdyn indisponible");
  case TypesSolver::ILUp    : throw ArgumentException("AlephMatrixPETSc", "preconditionnement ILUp indisponible");
  case TypesSolver::POLY    : throw ArgumentException("AlephMatrixPETSc", "preconditionnement POLY indisponible");
  default : throw ArgumentException("AlephMatrixPETSc", "preconditionnement inconnu");
  }

  if (solver_param->xoUser())
    KSPSetInitialGuessNonzero(m_ksp_solver,PETSC_TRUE);

  KSPSetTolerances(m_ksp_solver,
                   solver_param->epsilon(),
                   PETSC_DEFAULT,
                   PETSC_DEFAULT,
                   solver_param->maxIter());

  // override KSP options from commandline if present//
  KSPSetFromOptions(m_ksp_solver);

  // override PC options from commandline if present//
  PCSetFromOptions(prec) ;

  // setup KSP solver and precondtioner //
  KSPSetUp(m_ksp_solver);
  PCSetUp(prec);

  // print info which KSP solver and PC used //
  KSPType kt;
  KSPGetType(m_ksp_solver,&kt);
  PCType pt;
  PCGetType(prec,&pt);
  info(4) << "[AlephMatrixSolve] Will use KSP type :" << kt << " PC type : "<< pt
          << " is_parallel=" << is_parallel;

  // solve the linear system //
  debug()<<"[AlephMatrixSolve] All set up, now solving";
  petscCheck(KSPSolve(m_ksp_solver,RHS,solution));
  debug()<<"[AlephMatrixSolve] solved";
  petscCheck(KSPGetConvergedReason(m_ksp_solver,&reason));

  switch(reason){
#if !PETSC_VERSION_(3,0,0)
#if PETSC_VERSION_GE(3,24,0)
  case(KSP_CONVERGED_RTOL_NORMAL_EQUATIONS):{break;}
  case(KSP_CONVERGED_ATOL_NORMAL_EQUATIONS):{break;}
#else
  case(KSP_CONVERGED_RTOL_NORMAL):{break;}
  case(KSP_CONVERGED_ATOL_NORMAL):{break;}
#endif
#endif
  case(KSP_CONVERGED_RTOL):{break;}
  case(KSP_CONVERGED_ATOL):{break;}
  case(KSP_CONVERGED_ITS):{break;}
#if !PETSC_VERSION_GE(3,19,0)
  case(KSP_CONVERGED_CG_NEG_CURVE):{break;}
  case(KSP_CONVERGED_CG_CONSTRAINED):{break;}
#endif
  case(KSP_CONVERGED_STEP_LENGTH):{break;}
  case(KSP_CONVERGED_HAPPY_BREAKDOWN):{break;}
              /* diverged */
  case(KSP_DIVERGED_NULL):{ throw Exception("AlephMatrixPETSc::Solve", "KSP_DIVERGED_NULL");}
  case(KSP_DIVERGED_ITS):{ throw Exception("AlephMatrixPETSc::Solve", "KSP_DIVERGED_ITS");}
  case(KSP_DIVERGED_DTOL):{ throw Exception("AlephMatrixPETSc::Solve", "KSP_DIVERGED_DTOL");}
  case(KSP_DIVERGED_BREAKDOWN):{ throw Exception("AlephMatrixPETSc::Solve", "KSP_DIVERGED_BREAKDOWN");}
  case(KSP_DIVERGED_BREAKDOWN_BICG):{ throw Exception("AlephMatrixPETSc::Solve", "KSP_DIVERGED_BREAKDOWN_BICG");}
  case(KSP_DIVERGED_NONSYMMETRIC):{ throw Exception("AlephMatrixPETSc::Solve", "KSP_DIVERGED_NONSYMMETRIC");}
  case(KSP_DIVERGED_INDEFINITE_PC):{ throw Exception("AlephMatrixPETSc::Solve", "KSP_DIVERGED_INDEFINITE_PC");}
#if PETSC_VERSION_GE(3,6,1)
  case(KSP_DIVERGED_NANORINF):{ throw Exception("AlephMatrixPETSc::Solve", "KSP_DIVERGED_NANORINF");}
#else
  case(KSP_DIVERGED_NAN):{ throw Exception("AlephMatrixPETSc::Solve", "KSP_DIVERGED_NAN");}
#endif
  case(KSP_DIVERGED_INDEFINITE_MAT):{ throw Exception("AlephMatrixPETSc::Solve", "KSP_DIVERGED_INDEFINITE_MAT");}

  case(KSP_CONVERGED_ITERATING):{break;}
  default: throw Exception("AlephMatrixPETSc::Solve", "");
  }

  KSPGetIterationNumber(m_ksp_solver,&its);
  nb_iteration=its;

  KSPGetResidualNorm(m_ksp_solver,&norm);
  *residual_norm=norm;

  return 0;
}


// ****************************************************************************
// * writeToFile
// ****************************************************************************

void AlephMatrixPETSc::
writeToFile(const String filename)
{
  ARCANE_UNUSED(filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(PETScAlephFactoryImpl,IAlephFactoryImpl,PETScAlephFactory);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
