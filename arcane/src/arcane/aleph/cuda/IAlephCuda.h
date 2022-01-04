// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IAlephCnc.h                                                 (C) 2000-2012 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef _ALEPH_INTERFACE_CNC_H_
#define _ALEPH_INTERFACE_CNC_H_

#include "arcane/aleph/cuda/AlephCuda.h"

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
 

/******************************************************************************
 * AlephVectorCnc
 *****************************************************************************/
class AlephVectorCnc: public IAlephVector{
public:
AlephVectorCnc(ITraceMng *tm,
                               AlephKernel *kernel,
                               Integer index):IAlephVector(tm,kernel,index),
                                              m_cnc_vector(0){
  debug()<<"\t\t[AlephVectorCnc::AlephVectorCnc] new AlephVectorCnc";
} 
  
/******************************************************************************
 *****************************************************************************/
void AlephVectorCreate(void){
  debug()<<"\t\t[AlephVectorCnc::AlephVectorCreate] CNC VectorCreate 0-"<<m_kernel->topology()->nb_row_size()-1;
  m_cnc_vector.allocate(m_kernel->topology()->nb_row_size());
  debug()<<"\t\t[AlephVectorCnc::AlephVectorCreate] done";
}

/******************************************************************************
 *****************************************************************************/
void AlephVectorSet(const double *bfr_val, const int *bfr_idx, Integer size){
  debug()<<"\t\t[AlephVectorCnc::AlephVectorSet]";
  for(Integer i=0; i<size; ++i) {
    m_cnc_vector[bfr_idx[i]] = bfr_val[i];
  }
}

/******************************************************************************
 *****************************************************************************/
int AlephVectorAssemble(void){
  return 0; 
}

/******************************************************************************
 *****************************************************************************/
void AlephVectorGet(double *bfr_val, const int *bfr_idx, Integer size){
  debug()<<"\t\t[AlephVectorCnc::AlephVectorGet]";
  for(Integer i=0; i<size; ++i) {
    bfr_val[i]=m_cnc_vector[bfr_idx[i]];
  }
}

/******************************************************************************
 *****************************************************************************/
void writeToFile(const String filename){
  debug()<<"\t\t[AlephVectorCnc::writeToFile]";
  m_cnc_vector.print();
}
  
public:
  CNC_Vector<double> m_cnc_vector;
}; 


/******************************************************************************
 AlephMatrixCnc
 *****************************************************************************/
class AlephMatrixCnc: public IAlephMatrix{
public:


/******************************************************************************
 AlephMatrixCnc
*****************************************************************************/
AlephMatrixCnc(ITraceMng *tm,
               AlephKernel *kernel,
               Integer index):IAlephMatrix(tm,kernel,index),
                              m_cnc_matrix(0),
                              m_cuda(),
                              m_smcrs(),
                              has_been_allocated_on_gpu(false){
  debug()<<"\t\t[AlephMatrixCnc::AlephMatrixCnc] new SolverMatrixCnc";
}


/******************************************************************************
 * ~AlephMatrixCnc
 *****************************************************************************/
~AlephMatrixCnc(){
  debug()<<"\t\t[AlephMatrixCnc::~AlephMatrixCnc] DELETE SolverMatrixCnc";
  m_smcrs.gpu_deallocate();
}


/******************************************************************************
 *****************************************************************************/
void AlephMatrixCreate(void){
  debug()<<"\t\t[AlephMatrixCnc::AlephMatrixCreate] CNC MatrixCreate";
  m_cnc_matrix.allocate(m_kernel->topology()->nb_row_size(),
                        m_kernel->topology()->nb_row_size(),
                        CNC_Matrix::ROWS,
                        false); // symmetric ?
  m_cuda.cnc_cuda_set_dim_vec_from_n(m_kernel->topology()->nb_row_size());
}

/******************************************************************************
 *****************************************************************************/
void AlephMatrixSetFilled(bool){}

/******************************************************************************
 *****************************************************************************/
int AlephMatrixAssemble(void){
  debug()<<"\t\t[AlephMatrixCnc::AlephMatrixConfigure] AlephMatrixConfigure";
  m_cuda.convert_matrix(m_cnc_matrix, m_smcrs, false);
  if (has_been_allocated_on_gpu==false){
    m_smcrs.gpu_allocate();
    has_been_allocated_on_gpu=true;
  }
  return 0;
}

/******************************************************************************
 *****************************************************************************/
void AlephMatrixFill(int nrows, int *rows, int *cols, double *values){
  debug()<<"\t\t[AlephMatrixCnc::AlephMatrixFill]";
  for(int i=0,iMax=m_kernel->topology()->gathered_nb_setValued(m_kernel->rank());i<iMax;++i){
    m_cnc_matrix.add(rows[i],
                     cols[i],
                     values[i]);
  }
  debug()<<"\t\t[AlephMatrixCnc::AlephMatrixFill] done";
}
  
  /******************************************************************************
   * LinftyNormVectorProductAndSub
   *****************************************************************************/
  Real LinftyNormVectorProductAndSub(AlephVector* x,
                                     AlephVector* b){
    throw FatalErrorException("LinftyNormVectorProductAndSub", "error");
  }


/******************************************************************************
 *****************************************************************************/
int AlephMatrixSolve(AlephVector* x,
                     AlephVector* b,
                     AlephVector* t,
                     Integer& nb_iteration,
                     Real* residual_norm,
                     AlephParams* solver_param){
  const String func_name("SolverMatrixCnc::solve");
  
  debug()<<"\t\t[AlephMatrixCnc::AlephMatrixSolve] Getting X & B";
  CNC_Vector<double> *X = &(dynamic_cast<AlephVectorCnc*> (x->implementation()))->m_cnc_vector;
  CNC_Vector<double> *B = &(dynamic_cast<AlephVectorCnc*> (b->implementation()))->m_cnc_vector;

  debug()<<"\t\t[AlephMatrixCnc::AlephMatrixSolve] Flushing X";
  X->set_all(0.0f);
  
  // résolution du système algébrique
  unsigned int cnc_nb_iter_max = solver_param->maxIter();
  double cnc_epsilon = solver_param->epsilon();
  
  debug()<<"\t\t[AlephMatrixCnc::AlephMatrixSolve] Launching CNC_Solver";
  m_cuda.solve(m_smcrs, *B, *X, cnc_nb_iter_max, cnc_epsilon, nb_iteration, residual_norm);
  
/*  debug()<<"\t\t[AlephMatrixCnc::AlephMatrixSolve] HARD-CODING arguments";
  nb_iteration  = cnc_nb_iter_max-1;
  residual_norm[0] = cnc_epsilon;
*/
  
  if ( nb_iteration == solver_param->maxIter() && solver_param->stopErrorStrategy()){
	 info() << "\n============================================================";
	 info() << "\nCette erreur est retournée après " << nb_iteration << "\n";
	 info() << "\nOn a atteind le nombre max d'itérations du solveur.";
	 info() << "\nIl est possible de demander au code de ne pas tenir compte de cette erreur.";
	 info() << "\nVoir la documentation du jeu de données concernant le service solveur.";
	 info() << "\n======================================================";
	 throw  Exception("AlephMatrixCnc::Solve", "On a atteind le nombre max d'itérations du solveur");
  }
  return 0;
}
  
/******************************************************************************
 *****************************************************************************/
void writeToFile(const String filename){
  //CNC_MatrixPrint(m_cnc_matrix, filename.localstr());
}

public:
  CNC_Matrix m_cnc_matrix;
  Cuda m_cuda;
  CNC_MatrixCRS<double> m_smcrs;
  bool has_been_allocated_on_gpu;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // _ALEPH_INTERFACE_CNC_H_
