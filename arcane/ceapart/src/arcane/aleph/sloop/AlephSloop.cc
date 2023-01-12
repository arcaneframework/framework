// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephSloop.cc                                               (C) 2010-2023 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define HASMPI
#define SLOOP_MATH 
#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#define PARALLEL_SLOOP
#include <mpi.h>
#include "SLOOP.h"

#ifndef ItacFunction
#define ItacFunction(x)
#endif

#include "arcane/utils/FatalErrorException.h"
#include "arcane/aleph/AlephArcane.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


// ****************************************************************************
// * Class AlephTopologySloop
// ****************************************************************************
class AlephTopologySloop: public IAlephTopology{
public:
  AlephTopologySloop(ITraceMng* tm,
                     AlephKernel *kernel,
                     Integer index,
                     Integer nb_row_size):
    IAlephTopology(tm, kernel, index, nb_row_size),
    m_sloop_comminfo(NULL),
    m_world_comminfo(NULL),
    m_sloop_msg(NULL),
    m_sloop_topology(NULL){
    ItacFunction(AlephTopologySloop);

    if (!m_kernel->isParallel()){
      m_world_comminfo=new SLOOP::SLOOPSeqCommInfo();
    }else{
      m_world_comminfo=new SLOOP::SLOOPMPICommInfo(MPI_COMM_WORLD);
    }

    if (!m_participating_in_solver){
      debug() << "\33[1;32m\t[AlephTopologySloop::AlephParallelInfoSloop] Not concerned with this solver, returning\33[0m";
      return;    
    }
    
    debug() << "\33[1;32m\t\t[AlephTopologySloop::AlephTopologySloop] @"<<this<<"\33[0m";

    if (!m_kernel->isParallel()){
      m_sloop_comminfo=new SLOOP::SLOOPSeqCommInfo();
      debug() << "\33[1;32m\t\t[AlephTopologySloop::AlephTopologySloop] SEQCommInfo @"<<m_sloop_comminfo<<"\33[0m";
    }else{
      m_sloop_comminfo=new SLOOP::SLOOPMPICommInfo(*(SLOOP::SLOOP_Comm*)
                                                   (m_kernel->subParallelMng(index)->getMPICommunicator()));
      debug() << "\33[1;32m\t\t[AlephTopologySloop::AlephTopologySloop] MPICommInfo @"<<m_sloop_comminfo<<"\33[0m";
    }

    m_sloop_msg=new SLOOP::SLOOPMsg(m_sloop_comminfo);
    debug() << "\33[1;32m\t\t[AlephTopologySloop::AlephTopologySloop] SLOOPMsg @"<<m_sloop_msg<<"\33[0m";

    m_sloop_topology=new SLOOP::SLOOPTopology(nb_row_size,                                            
                                              m_kernel->topology()->rowLocalRange(index),
                                              *m_sloop_comminfo, *m_sloop_msg);
    debug() << "\33[1;32m\t\t[AlephTopologySloop::AlephTopologySloop] SLOOPTopology @"<<m_sloop_topology<<"\33[0m";
  }
  ~AlephTopologySloop(){
    debug() << "\33[1;5;32m\t\t\t[~AlephTopologySloop] deleting m_sloop_msg, m_sloop_comminfo & m_sloop_topology\33[0m";
    delete m_sloop_msg;
    delete m_sloop_comminfo;
    delete m_sloop_topology;
  }
public:
  // On backup la session courante et on initialise la nouvelle session avec notre m_sloop_comminfo
  void backupAndInitialize(){
    if (!m_participating_in_solver) return;
    SLOOP::SLOOPInitSession(m_sloop_comminfo,SLOOP::WITHOUT_EXCEPTIONS,SLOOP::OUTPUT_UNIQ);
  }
  void restore(){
    // Dans tous les cas, tout le monde restore la WORLD session
    SLOOP::SLOOPInitSession(m_world_comminfo,SLOOP::WITHOUT_EXCEPTIONS,SLOOP::OUTPUT_UNIQ);
    
  }
public:
  SLOOP::SLOOPCommInfo *m_sloop_comminfo;
  SLOOP::SLOOPCommInfo *m_world_comminfo;
  SLOOP::SLOOPMsg *m_sloop_msg;
  SLOOP::SLOOPTopology *m_sloop_topology;
};


// ****************************************************************************
// * AlephVectorSloop
// ****************************************************************************
class AlephVectorSloop
: public IAlephVector
{
 public:
  AlephVectorSloop(ITraceMng* tm,
                   AlephKernel *kernel,
                   Integer index):IAlephVector(tm,kernel,index),
                                  m_sloop_vector(NULL){
    debug()<<"\t\t[AlephVectorSloop::AlephVectorSloop] NEW AlephVectorSloop";
  }
  
  // ****************************************************************************
  // * ~AlephVectorSloop
  // ****************************************************************************
  ~AlephVectorSloop(){
    debug() << "\33[1;5;32m\t\t\t[~AlephVectorSloop]\33[0m";
    delete m_sloop_vector;
  }
  
  /******************************************************************************
   * AlephVectorCreate
   *****************************************************************************/
  void AlephVectorCreate(void)
  {
    ItacFunction(AlephVectorSloop);
    debug()<<"\t[AlephVectorSloop::AlephVectorCreate] new SLOOP::SLOOPDistVector";
    AlephTopologySloop* aleph_topology_sloop = dynamic_cast<AlephTopologySloop*>(m_kernel->getTopologyImplementation(m_index));
    ARCANE_CHECK_POINTER(aleph_topology_sloop);
    m_sloop_vector = new SLOOP::SLOOPDistVector(*aleph_topology_sloop->m_sloop_topology,
                                                *aleph_topology_sloop->m_sloop_msg);
    if (!m_sloop_vector)
      throw FatalErrorException(A_FUNCINFO, " new SLOOPDistVector failed");
    debug()<<"\t[AlephVectorSloop::AlephVectorCreate] done";
  }

  // ****************************************************************************
  // * AlephVectorSet
  // ****************************************************************************
  void AlephVectorSet(const double *bfr_val, const int *bfr_idx, Integer size)
  {
    debug()<<"\t[AlephVectorSloop::AlephVectorSet]";
    if (m_sloop_vector->locfill(bfr_val, bfr_idx, size))
      throw FatalErrorException(A_FUNCINFO, "locfill() failed");
  }

/******************************************************************************
 * AlephVectorAssemble
 *****************************************************************************/
int AlephVectorAssemble(void){
  ItacFunction(AlephVectorSloop);
  debug()<<"\t\t[AlephVectorSloop::AlephVectorAssemble]";
  return 0;
}


/******************************************************************************
 * AlephVectorGet
 *****************************************************************************/
void AlephVectorGet(double *bfr_val, const int *bfr_idx, Integer size){
  // Je récupère déjà les valeurs depuis le vecteur Sloop
  debug()<<"\t[AlephVectorSloop::AlephVectorGet]";
  if (m_sloop_vector->get_locval(bfr_val, bfr_idx, size))
	 throw Exception("AlephVectorSloop::AlephVectorGet", "get_locval() failed");
}


/******************************************************************************
 * writeToFile
 *****************************************************************************/
  void writeToFile(const String file_name){
    ItacFunction(AlephVectorSloop);
    m_sloop_vector->write_to_file(file_name.localstr());
  }
  
public:
  SLOOP::SLOOPDistVector* m_sloop_vector;
};




/******************************************************************************
 AlephMatrixSloop
*****************************************************************************/
class AlephMatrixSloop: public IAlephMatrix{
public:

/******************************************************************************
 * AlephMatrixSloop
 *****************************************************************************/
  AlephMatrixSloop(ITraceMng* tm,
                   AlephKernel *kernel,
                   Integer index):IAlephMatrix(tm,kernel,index),
                                  m_sloop_matrix(NULL){
    debug()<<"\t\t[AlephMatrixSloop::AlephMatrixSloop] NEW AlephMatrixSloop";
  }
  
// ****************************************************************************
// * AlephMatrixSloop
// ****************************************************************************
  ~AlephMatrixSloop(){
    debug() << "\33[1;5;32m\t\t\t[~AlephMatrixSloop]\33[0m";
    delete m_sloop_matrix;
  }


  
  /******************************************************************************
   * AlephMatrixCreate
   *****************************************************************************/
  void AlephMatrixCreate(void)
  {
    ItacFunction(AlephMatrixSloop);
    debug()<<"\t\t[AlephMatrixSloop::AlephMatrixCreate] create new SLOOP::SLOOPDistMatrix";
    AlephTopologySloop* aleph_topology_sloop = dynamic_cast<AlephTopologySloop*>(m_kernel->getTopologyImplementation(m_index));
    ARCANE_CHECK_POINTER(aleph_topology_sloop);
    m_sloop_matrix = new SLOOP::SLOOPDistMatrix(*aleph_topology_sloop->m_sloop_topology,
                                                *aleph_topology_sloop->m_sloop_msg,
                                                true,
                                                false);
    
    if (!m_sloop_matrix) throw Exception("AlephSolverMatrix::create","new SLOOPDistMatrix() failed");
    
    // la renumerotation doit etre faite avant le remplissage de la matrice pour etre prise en compte
    if (aleph_topology_sloop->m_sloop_topology->get_type()==SLOOP::contiguous)
      m_sloop_matrix->set_renumbering_opt(SLOOP::SLOOPDistMatrix::interface, false);
    m_sloop_matrix->set_renumbering_opt(SLOOP::SLOOPDistMatrix::processor, false);
    m_sloop_matrix->set_renumbering_opt(SLOOP::SLOOPDistMatrix::interior, false);
    m_sloop_matrix->init_length(m_kernel->topology()->gathered_nb_row_elements().unguardedBasePointer());
    debug()<<"\t\t[AlephMatrixSloop::AlephMatrixCreate] done";
  }
  
  
/******************************************************************************
 * AlephMatrixSetFilled
 *****************************************************************************/
  void AlephMatrixSetFilled(bool toggle){
    ItacFunction(AlephMatrixSloop);
    debug()<<"\t\t[AlephMatrixSloop::AlephMatrixSetFilled]";
    m_sloop_matrix->setfilled(toggle);
  }


/******************************************************************************
 * AlephMatrixAssemble
 *****************************************************************************/
  int AlephMatrixAssemble(void){
    ItacFunction(AlephMatrixSloop);
    debug()<<"\t\t[AlephMatrixSloop::AlephMatrixAssemble]";
    m_sloop_matrix->configure();
    return 0;
  }

  
/******************************************************************************
 * AlephMatrixFill
 *****************************************************************************/
  void AlephMatrixFill(int size, int *rows, int *cols, double *values){
    ItacFunction(AlephMatrixSloop);
    debug()<<"\t\t[AlephMatrixSloop::AlephMatrixFill] size="<<size;
    m_sloop_matrix->locfill(values, rows, cols, size);
    debug()<<"\t\t[AlephMatrixSloop::AlephMatrixFill] done";
  }

  
/******************************************************************************
 * isAlreadySolved
 *****************************************************************************/
  bool isAlreadySolved(SLOOP::SLOOPDistVector* x,
                       SLOOP::SLOOPDistVector* b,
                       SLOOP::SLOOPDistVector* tmp,
                       Real* residual_norm,
                       AlephParams* params) {
    const Real res0 = b->norm_max();
    const Real considered_as_null = params->minRHSNorm();
    const bool convergence_analyse = params->convergenceAnalyse();
	
    if (convergence_analyse)
      debug() << "analyse convergence : norme max du second membre res0 : " << res0;
   
    if (res0 < considered_as_null) {
		x->fill(Real(0.0));
		residual_norm[0]= res0;
		if (convergence_analyse)
        debug() << "analyse convergence : le second membre du système linéaire est inférieur à "
                << considered_as_null;
		return true;
    }

    if (params->xoUser()) {
      // on test si b est déjà solution du système à epsilon près
      m_sloop_matrix->vector_multiply(*tmp,*x);  // tmp=A*x
      tmp->substract(*tmp,*b);                   // tmp=A*x-b
      const Real residu= tmp->norm_max(); 
      debug() << "[IAlephSloop::isAlreadySolved] residu="<<residu;

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
    return false;
  }



  
  /******************************************************************************
   *****************************************************************************/
  int AlephMatrixSolve(AlephVector* x,
                       AlephVector* b,
                       AlephVector* tmp,
                       Integer& nb_iteration,
                       Real* residual_norm,
                       AlephParams* solver_param)
  {
    ItacFunction(AlephMatrixSloop);   
    Integer status = 0;
    debug() << "\t\t[AlephMatrixSloop::SloopSolve] getTopologyImplementation #"<<m_index;
    AlephTopologySloop* sloopParallelInfo = dynamic_cast<AlephTopologySloop*>(m_kernel->getTopologyImplementation(m_index));
    
    AlephVectorSloop* x_sloop = dynamic_cast<AlephVectorSloop*> (x->implementation());
    AlephVectorSloop* b_sloop = dynamic_cast<AlephVectorSloop*> (b->implementation());
    AlephVectorSloop* tmp_sloop = dynamic_cast<AlephVectorSloop*> (tmp->implementation());

    ARCANE_CHECK_POINTER(x_sloop);
    ARCANE_CHECK_POINTER(b_sloop);
    ARCANE_CHECK_POINTER(tmp_sloop);

    SLOOP::SLOOPDistVector* solution = x_sloop->m_sloop_vector;
    SLOOP::SLOOPDistVector* RHS      = b_sloop->m_sloop_vector;
    SLOOP::SLOOPDistVector* temp     = tmp_sloop->m_sloop_vector;
    
    if (isAlreadySolved(solution,RHS,temp,residual_norm,solver_param)){
      debug() << "\t[AlephMatrixSloop::AlephMatrixSolve] isAlreadySolved !";
      nb_iteration = 0;
      return 0;
    }
    
    ScopedPtrT<SLOOP::SLOOPStopCriteria> sc;
    sc=createSloopStopCriteria(solver_param, *sloopParallelInfo->m_sloop_msg);
    
    ScopedPtrT<SLOOP::SLOOPSolver> global_solver;
    global_solver=createSloopSolver(solver_param, *sloopParallelInfo->m_sloop_msg);
    
    ScopedPtrT<SLOOP::SLOOPPreconditioner> precond;
    precond=createSloopPreconditionner(solver_param, *sloopParallelInfo->m_sloop_msg);
    
    this->setSloopSolverParameters(solver_param, global_solver.get());
    this->setSloopPreconditionnerParameters(solver_param, precond.get());
    
    ScopedPtrT<SLOOP::SLOOPDistVector> diag;
    const bool normalize = normalizeSolverMatrix(solver_param);
    if (normalize) {
      debug() << "\t\t[AlephMatrixSloop::SloopSolve] Normalize";
      diag = new SLOOP::SLOOPDistVector(*sloopParallelInfo->m_sloop_topology, *sloopParallelInfo->m_sloop_msg);
      global_solver->normalize(*m_sloop_matrix, *diag, *solution, *RHS);
    }
    
    const bool xo = solver_param->xoUser();
    switch (solver_param->precond()) {
    case TypesSolver::NONE:
      // appel sans preconditionnement : utilise dans le CAS des solveurs SAMG et SuperLU
      if (xo){
        debug() << "\t\t[AlephMatrixSloop::SloopSolve] xo à true (sans preconditionnement)";
        status = global_solver->solve(*m_sloop_matrix, *solution, *RHS, *sc);
      }else{
        debug() << "\t\t[AlephMatrixSloop::SloopSolve] xo à false (sans preconditionnement)";
        status = global_solver->solve_b(*m_sloop_matrix, *solution, *RHS, *sc);
      }
      break;
    default:
      // appel avec preconditionnement
      if (xo){
        debug() << "\t\t[AlephMatrixSloop::SloopSolve] xo à true (avec preconditionnement)";
        status = global_solver->solve(*m_sloop_matrix, *solution, *RHS, *precond, *sc);
      } else{
        debug() << "\t\t[AlephMatrixSloop::SloopSolve] xo à false (avec preconditionnement)";
        status = global_solver->solve_b(*m_sloop_matrix, *solution, *RHS, *precond, *sc);
      }
      break;
    }
    
    if (normalize) {
      debug() << "\t\t[AlephMatrixSloop::SloopSolve] INV-normalize";
      global_solver->inv_normalize(*m_sloop_matrix, *diag, *solution, *RHS);
    }
    
    nb_iteration = global_solver->get_iteration();
    residual_norm[0] = sc->get_criteria();
    Integer max_iteration= global_solver->get_max_iteration();
    residual_norm[3] = global_solver->get_stagnation();
    
    if ((solver_param->getCriteriaStop()==TypesSolver::STAG)||(solver_param->getCriteriaStop()==TypesSolver::NIter)){
      // pas de controle des iterations dans les cas du critere de stagnation
      // et du critere sur le nombre d'iterations impose du solveur
    }else{
      if (nb_iteration == max_iteration && solver_param->stopErrorStrategy()){
        info() << "\n============================================================";
        info() << "\nCette erreur est retournée après " << nb_iteration << "\n";
        info() << "\nOn a atteind le nombre max d'itérations du solveur ";
        info() << "\nIl est possible de demander au code de ne pas tenir compte de cette erreur.";
        info() << "\nVoir la documentation du jeu de données concernant le service solveur.";
        info() << "\n============================================================";
        throw Exception("AlephMatrixSloop::SloopSolve", "On a atteind le nombre max d'itérations du solveur");
      }
    }
    debug() << "\t\t[AlephMatrixSloop::SloopSolve] nbIteration="<<  global_solver->get_iteration()
            << ", criteria=" << residual_norm[0] << ", stagnation=" << residual_norm[3];
    return status;
  }
  
  
/******************************************************************************
 * writeToFile
 *****************************************************************************/
  void writeToFile(const String file_name){
    ItacFunction(AlephMatrixSloop);
    m_sloop_matrix->write_to_file(file_name.localstr());
  }

  
/******************************************************************************
 * createSloopSolver
 *****************************************************************************/
  SLOOP::SLOOPSolver* createSloopSolver(AlephParams* solver_param, SLOOP::SLOOPMsg& info_sloop_msg){
    const Integer output_level = solver_param->getOutputLevel();
    ItacFunction(AlephMatrixSloop);
    switch (solver_param->method()) {
    case TypesSolver::PCG:       return new SLOOP::SLOOPCGSolver(info_sloop_msg, output_level);
    case TypesSolver::BiCGStab:  return new SLOOP::SLOOPBiCGStabSolver(info_sloop_msg, output_level);
    case TypesSolver::BiCGStab2: return new SLOOP::SLOOPBiCGStabSolver(info_sloop_msg, output_level);
    case TypesSolver::GMRES:     return new SLOOP::SLOOPGMRESSolver(info_sloop_msg, output_level);
    case TypesSolver::SAMG:      return new SLOOP::SLOOPAMGSolver(info_sloop_msg, output_level);
    case TypesSolver::QMR:       return new SLOOP::SLOOPQMRSolver(info_sloop_msg, output_level);
    case TypesSolver::SuperLU:{
      SLOOP::SLOOPSolver *solver = new SLOOP::SLOOPSuperLUSolver(info_sloop_msg, output_level);
      solver->set_parameter(SLOOP::sv_residual_calculation, 1);
      return solver;
    }
    default: throw Exception("AlephMatrixSloop::createSloopSolver", "Type de solver non accessible pour la bibliothèque SLOOP");
    }
    return NULL;
  }


/******************************************************************************
 * createSloopPreconditionner
 *****************************************************************************/
  SLOOP::SLOOPPreconditioner* createSloopPreconditionner(AlephParams* solver_param,
                                                         SLOOP::SLOOPMsg& info_sloop_msg){
    const Integer output_level = solver_param->getOutputLevel();
    ItacFunction(AlephMatrixSloop);
    switch (solver_param->precond())	{
    case TypesSolver::AINV: 
      // Pour le moment utilisation AINV(0) pour des matrices symetriques
      // et SPAI pour les matrices non symetriques
      if (TypesSolver::PCG == solver_param->method())
        return new SLOOP::SLOOPAINVPC(info_sloop_msg, output_level);
      else
        return new SLOOP::SLOOPMAINVPC(info_sloop_msg, output_level);
    case TypesSolver::DIAGONAL: return new SLOOP::SLOOPDiagPC(info_sloop_msg, output_level);
    case TypesSolver::AMG:      return new SLOOP::SLOOPAMGPC(info_sloop_msg, output_level);
    case TypesSolver::IC:       return new SLOOP::SLOOPCholPC(info_sloop_msg, output_level);
    case TypesSolver::POLY:     return new SLOOP::SLOOPPolyPC(info_sloop_msg, output_level);
    case TypesSolver::ILU:      return new SLOOP::SLOOPILUPC(info_sloop_msg, output_level);
    case TypesSolver::ILUp:     return new SLOOP::SLOOPILUPC(info_sloop_msg, output_level);
    case TypesSolver::SPAIstat: return new SLOOP::SLOOPSPAIPC(info_sloop_msg, output_level);
    case TypesSolver::SPAIdyn:  return new SLOOP::SLOOPSPAIPC(info_sloop_msg, output_level);
    case TypesSolver::NONE:     return NULL;
    default: throw Exception("AlephMatrixSloop::createSloopPreconditionner",
                             "préconditionneur non accessible pour la bibliothèque SLOOP");
    }
    return NULL;
  }


/******************************************************************************
 * createSloopStopCriteria
 *****************************************************************************/
  SLOOP::SLOOPStopCriteria * createSloopStopCriteria(AlephParams* solver_param,
                                                     SLOOP::SLOOPMsg& info_sloop_msg) {
    ItacFunction(AlephMatrixSloop);
    switch(solver_param->getCriteriaStop()){
    case TypesSolver::RR0:    return new SLOOP::SLOOPStopCriteriaRR0();
    case TypesSolver::R:      return new SLOOP::SLOOPStopCriteriaR();
    case TypesSolver::RCB:    return new SLOOP::SLOOPStopCriteriaRCB();
    case TypesSolver::RBinf:  return new SLOOP::SLOOPStopCriteriaRBinf();
    case TypesSolver::EpsA:   return new SLOOP::SLOOPStopCriteriaEpsA();
    case TypesSolver::NIter:  return new SLOOP::SLOOPStopCriteriaNIter();
    case TypesSolver::RR0inf: return new SLOOP::SLOOPStopCriteriaRR0inf();
    case TypesSolver::STAG:   return new SLOOP::SLOOPStopCriteriaSTAG();
    case TypesSolver::RB:
    default: return new SLOOP::SLOOPStopCriteriaRB();
    }
    return NULL;
  }


/******************************************************************************
 *****************************************************************************/
  void setSloopSolverParameters(AlephParams* solver_param,
                                SLOOP::SLOOPSolver* sloop_solver){
    Integer error=0;
    Integer gamma = solver_param->gamma();
    double alpha = solver_param->alpha();
    ItacFunction(AlephMatrixSloop);
    error+=sloop_solver->set_parameter(SLOOP::sv_epsilon, solver_param->epsilon());
    error+=sloop_solver->set_parameter(SLOOP::sv_max_iter, solver_param->maxIter());
    switch (solver_param->method()) {
    case TypesSolver::PCG: break; // TODOMB  valeurs propres //error += sloop_solver->set_parameter(cg_spectrum_size, 50);
    case TypesSolver::QMR: break;
    case TypesSolver::SuperLU: break;
    case TypesSolver::BiCGStab: error += sloop_solver->set_parameter(SLOOP::bicg_dimension, 1); break;
    case TypesSolver::BiCGStab2: error += sloop_solver->set_parameter(SLOOP::bicg_dimension, 2); break;
    case TypesSolver::GMRES:
      error += sloop_solver->set_parameter(SLOOP::gmres_order, 20);
      error += sloop_solver->set_parameter(SLOOP::gmres_type, SLOOP::ICGS);
      break;
    case TypesSolver::SAMG:	{
      error += sloop_solver->set_parameter(SLOOP::amg_buffer_size, 200);
      // gamma niveaux maximum de deraffinement
      if (gamma == -1) gamma = 50; // valeur par defaut du nombre de niveaux
      error += sloop_solver->set_parameter(SLOOP::amg_level, gamma);
      //(solver_param->gamma() == -1)?50:solver_param->gamma());//Works fine here
      // valeur par defaut du parametre d'influence
      if (alpha < 0.0) alpha = 0.1;
      error += sloop_solver->set_parameter(SLOOP::amg_alpha, alpha);
      //(solver_param->alpha() < 0.0)?0.1:solver_param->alpha());// Works here too
      // nombre d'itérations du lisseur du solveur AMG
      error += sloop_solver->set_parameter(SLOOP::amg_smoother_iter, solver_param->getAmgSmootherIter());
      //1= cycle en V , 2=cycle en W, 3=cycle en FullMultigridV       (defaut 1)
      error += sloop_solver->set_parameter(SLOOP::amg_iter, solver_param->getAmgCycle());
      // definition du deraffinement AMG
      SLOOP::SLOOPAMGCoarseningOption coarsening_option =
        static_cast<SLOOP::SLOOPAMGCoarseningOption> (solver_param->getAmgCoarseningOption());
      error += sloop_solver->set_parameter(SLOOP::amg_coarsening_option, coarsening_option);
      //error += sloop_solver->set_parameter(SLOOP::amg_coarsening_option, solver_param->getAmgCoarseningOption());
      // definition du solveur du deraffinement
      SLOOP::SLOOPAMGCoarseSolverOption coarse_solver_option =
        static_cast<SLOOP::SLOOPAMGCoarseSolverOption> (solver_param->getAmgCoarseSolverOption());
      error += sloop_solver->set_parameter(SLOOP::amg_coarse_solver_option, coarse_solver_option);
      //error += sloop_solver->set_parameter(SLOOP::amg_coarse_solver_option, solver_param->getAmgCoarseSolverOption());
      // definition du lisseur AMG
      SLOOP::SLOOPAMGSmootherOption smoother_option =
        static_cast<SLOOP::SLOOPAMGSmootherOption> (solver_param->getAmgSmootherOption());
      error += sloop_solver->set_parameter(SLOOP::amg_smoother_option, smoother_option);
      //error += sloop_solver->set_parameter(SLOOP::amg_smoother_option, solver_param->getAmgSmootherOption());
    }
      break;
    default: throw Exception("AlephMatrixSloop::setSloopSolverParameters",
                             "Type de solver SLOOP non prévu dans la gestion des paramètres");
    }
    if (error)
      throw Exception("AlephMatrixSloop::setSloopSolverParameters", "set_parameter() failed");
  }
  

/******************************************************************************
 *****************************************************************************/
  void setSloopPreconditionnerParameters(AlephParams* solver_param,
                                         SLOOP::SLOOPPreconditioner* preconditionner){
    const TypesSolver::ePreconditionerMethod precond_method = solver_param->precond();
    double alpha = solver_param->alpha();
    int gamma = solver_param->gamma();
    const String function_id = "SolverMatrixSloop::setSloopPreconditionnerParameters";
    ItacFunction(AlephMatrixSloop);
    //valeur par defaut du parametre d'influence
    if (alpha < 0.0) alpha = 0.1; // 0.25 ;
    switch (precond_method) {
    case TypesSolver::NONE: break;
    case TypesSolver::DIAGONAL: break;
    case TypesSolver::AMG: {
      if (gamma == -1) gamma = 50;
      preconditionner->set_parameter(SLOOP::amg_level, gamma);//(solver_param->gamma()==-1)?50:solver_param->gamma());
      preconditionner->set_parameter(SLOOP::amg_buffer_size, 200);
      preconditionner->set_parameter(SLOOP::amg_solver_iter, solver_param->getAmgSolverIter());
      preconditionner->set_parameter(SLOOP::amg_iter, solver_param->getAmgCycle());
      preconditionner->set_parameter(SLOOP::amg_smoother_iter, solver_param->getAmgSmootherIter());
      // option du lisseur
      SLOOP::SLOOPAMGSmootherOption smoother_option =
        static_cast<SLOOP::SLOOPAMGSmootherOption> (solver_param->getAmgSmootherOption());
      // option du deraffinement
      SLOOP::SLOOPAMGCoarseningOption coarsening_option =
        static_cast<SLOOP::SLOOPAMGCoarseningOption> (solver_param->getAmgCoarseningOption());
      //  choix du solveur du systeme grossier  defaut (CG_coarse_solver) pour une matrice symetrique
      SLOOP::SLOOPAMGCoarseSolverOption coarse_solver_option =
        static_cast<SLOOP::SLOOPAMGCoarseSolverOption> (solver_param->getAmgCoarseSolverOption());
      // options pour matrice symetrique
      if (TypesSolver::PCG == solver_param->method()) {
        // controle du smoother  pour matrice symetrique
        switch (solver_param->getAmgSmootherOption()) {
        case TypesSolver::CG_smoother:
        case TypesSolver::Rich_IC_smoother:
        case TypesSolver::Rich_AINV_smoother:
        case TypesSolver::SymHybGSJ_smoother:
        case TypesSolver::Rich_IC_block_smoother:
        case TypesSolver::SymHybGSJ_block_smoother:
          break;
        default: throw Exception("AlephMatrixSloop::setSloopPreconditionnerParameters",
                                 "choix du smoother incorrect  pour une matrice symetrique");
        }
      }else{		//options pour matrices non symetrique
		  // choix du lisseur dans le cas non-symetrique
        switch (solver_param->getAmgSmootherOption()) {
        case TypesSolver::CG_smoother:
        case TypesSolver::Rich_IC_smoother:
        case TypesSolver::Rich_AINV_smoother:
        case TypesSolver::Rich_IC_block_smoother:
        case TypesSolver::SymHybGSJ_block_smoother:
          throw Exception("AlephMatrixSloop::setSloopPreconditionnerParameters",
                          "choix du smoother incorrect avec une matrice non-symetrique ");
        case TypesSolver::SymHybGSJ_smoother:
          // on modifie la valeur mise par defaut pour AMG
          solver_param->setAmgSmootherOption(TypesSolver::HybGSJ_smoother);
          break;
        default: break;
        }
        // controle du solveur
        switch (solver_param->getAmgCoarseSolverOption()) {
        case TypesSolver::CG_coarse_solver:
        case TypesSolver::Cholesky_coarse_solver:
          solver_param->setAmgCoarseSolverOption(TypesSolver::LU_coarse_solver);
          break;
        default:solver_param->setAmgCoarseSolverOption(TypesSolver::BiCGStab_coarse_solver); // choix du solveur du systeme grossier
          break;
        }
      } // fin du if matrice symetrique
		// definition du lisseur AMG apres controle
      preconditionner->set_parameter(SLOOP::amg_smoother_option, smoother_option);//solver_param->getAmgSmootherOption());
		// definition du deraffinement AMG
      preconditionner->set_parameter(SLOOP::amg_coarsening_option, coarsening_option);//solver_param->getAmgCoarseningOption());
      // definition du solveur du systeme grossier apres controle
      preconditionner->set_parameter(SLOOP::amg_coarse_solver_option, coarse_solver_option);//solver_param->getAmgCoarseSolverOption());
    }
      break;
	 
    case TypesSolver::POLY:
		if (gamma == -1) gamma = 3; // la valeur pour l'ordre du polynome
		break;
    case TypesSolver::AINV:
      if (gamma == -1) gamma = 0; // valeur par defaut pour le parametre de remplissage -> AINV0
      // le système linéaire doit être normalisé
      break;
    case TypesSolver::IC:
    case TypesSolver::ILU: if (gamma == -1) gamma = 0; break;
    case TypesSolver::ILUp: if (gamma == -1) gamma = 1; break;
    case TypesSolver::SPAIstat:
      preconditionner->set_parameter(SLOOP::spai_sparsity,SLOOP::StatSparsity);
      preconditionner->set_parameter(SLOOP::spai_init_sparsity,SLOOP::PowerSparsity);
      preconditionner->set_parameter(SLOOP::spai_power_level, 1);
      preconditionner->set_parameter(SLOOP::spai_Amax_row_size, 30);
      preconditionner->set_parameter(SLOOP::spai_A_drop_eps, 0.001);
      break;
    case TypesSolver::SPAIdyn:
      preconditionner->set_parameter(SLOOP::spai_sparsity, SLOOP::DynSparsity);
      preconditionner->set_parameter(SLOOP::spai_init_sparsity, SLOOP::DiagSparsity);
      preconditionner->set_parameter(SLOOP::spai_Amax_row_size, 10);
      preconditionner->set_parameter(SLOOP::spai_A_drop_eps, 0.001);
      break;
    default:
      throw Exception("AlephMatrixSloop::setSloopPreconditionnerParameters", "Préconditionneur inconnu.");
    }
  
    // initialisations communes aux preconditionneurs ( sauf sans precond )
    switch (precond_method) {
    case TypesSolver::NONE:
    case TypesSolver::SPAIstat:
    case TypesSolver::SPAIdyn:
      break;
	 
    case TypesSolver::AMG:
      preconditionner->set_parameter(SLOOP::amg_alpha, alpha);
      preconditionner->set_parameter(SLOOP::amg_level, gamma);
      preconditionner->set_parameter(SLOOP::amg_parallel_opt, 0);
      // definition du critere d'arret du solveur grossier (default RB)
      preconditionner->set_parameter(SLOOP::amg_coarse_solver_sc_option,SLOOP::RB);
      break;
	 
    default:
      preconditionner->set_parameter(SLOOP::pc_alpha, alpha);
      preconditionner->set_parameter(SLOOP::pc_nbelem, gamma);
      preconditionner->set_parameter(SLOOP::pc_parallel_opt, 1);
      preconditionner->set_parameter(SLOOP::pc_order, gamma);
      break;
    }
	 
  }
  

/******************************************************************************
 *****************************************************************************/
  bool normalizeSolverMatrix(AlephParams* solver_param){
    ItacFunction(AlephMatrixSloop);
    switch (solver_param->precond()){
    case TypesSolver::AINV:
      //case TypesSolver::AMG:
    case TypesSolver::SPAIstat:
    case TypesSolver::SPAIdyn: return true;
    case TypesSolver::AMG:
    case TypesSolver::NONE:
    case TypesSolver::DIAGONAL:
    case TypesSolver::IC:
    case TypesSolver::POLY:
    case TypesSolver::ILU:
    case TypesSolver::ILUp: return false;
    default: throw Exception("AlephMatrixSloop::normalizeSolverMatrix", "Préconditionneur inconnu.");
    }
    return false;
  }
public:
  SLOOP::SLOOPMatrix* m_sloop_matrix;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SloopAlephFactoryImpl: public AbstractService,
                             public IAlephFactoryImpl{
public:
  SloopAlephFactoryImpl(const ServiceBuildInfo& sbi) :
    AbstractService(sbi),
    m_IAlephVectors(0),
    m_IAlephMatrixs(0),
    m_IAlephTopologys(0){}
  ~SloopAlephFactoryImpl(){
    debug() << "\33[1;32m[~SloopAlephFactoryImpl]\33[0m";
    for(Integer i=0,iMax=m_IAlephVectors.size(); i<iMax; ++i)
      delete m_IAlephVectors.at(i);
    for(Integer i=0,iMax=m_IAlephMatrixs.size(); i<iMax; ++i)
      delete m_IAlephMatrixs.at(i);
    for(Integer i=0,iMax=m_IAlephTopologys.size(); i<iMax; ++i)
      delete m_IAlephTopologys.at(i);
  }
public:
  virtual void initialize() {}
  
  virtual IAlephTopology* createTopology(ITraceMng* tm,
                                         AlephKernel* kernel,
                                         Integer index,
                                         Integer nb_row_size){
    IAlephTopology *new_topology=new AlephTopologySloop(tm, kernel, index, nb_row_size);
    m_IAlephTopologys.add(new_topology);
    return new_topology;
  }
  
  virtual IAlephVector* createVector(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     Integer index){
    IAlephVector *new_vector=new AlephVectorSloop(tm,kernel,index);
    m_IAlephVectors.add(new_vector);
    return new_vector;
  }

  virtual IAlephMatrix* createMatrix(ITraceMng* tm,
                                     AlephKernel* kernel,
                                     Integer index){
    IAlephMatrix *new_matrix=new AlephMatrixSloop(tm,kernel,index);
    m_IAlephMatrixs.add(new_matrix);
    return new_matrix;
  }
private:
  UniqueArray<IAlephVector*> m_IAlephVectors;
  UniqueArray<IAlephMatrix*> m_IAlephMatrixs;
  UniqueArray<IAlephTopology*> m_IAlephTopologys;
};

ARCANE_REGISTER_APPLICATION_FACTORY(SloopAlephFactoryImpl,IAlephFactoryImpl,SloopAlephFactory);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
