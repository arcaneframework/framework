// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef _IALEPH_CUDA_H_
#define _IALEPH_CUDA_H_

#include <set>
#include <vector>
#include <algorithm>
#include <stdio.h>

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>

#define CNC_ASSERT(isOK, message)                                     \
  if (!(isOK)) {                                                      \
    (void)printf("ERROR!! Assert '%s' failed\n%s\n",                  \
                 #isOK, message);                                     \
    return false ;                                                    \
  }

//---------------------------------------------------------------------------//

template<class T, class S> inline S * CNCallocate_and_copy ( const T * in, int size ) {
		
  S * out = new S[size] ;
  for ( int i=0; i<size; i++ ) {
    out[i] = (S)in[i] ;
  }
  return out ;
}

//---------------------------------------------------------------------------//

template <class T> T* CNCallocate ( long number ) {
  return new T[number] ;
}
        
//---------------------------------------------------------------------------//

template <class T> void CNCdeallocate ( T*& addr ) {
  delete[] addr ;
  addr = NULL ; // makes it possible to track
  // access to deallocated memory.
}
        
//---------------------------------------------------------------------------//

template <class T> void CNCreallocate (T*& addr, long old_number, long new_number) {
  T* new_addr = new T[new_number] ;
  for(int i=0; i<old_number; i++) {
    new_addr[i] = addr[i] ;
  }
  delete[] addr ;
  addr = new_addr ;
}

#include "arcane/aleph/cuda/AlephCudaVector.h"
#include "arcane/aleph/cuda/AlephCudaMatrix.h"
#include "arcane/aleph/cuda/AlephCudaMatrixCrs.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Cuda{
public:
  Cuda();
  ~Cuda();
  void cnc_cuda_set_dim_vec_from_n(long);
  void convert_matrix(const CNC_Matrix& rhs, CNC_MatrixCRS<double>& A, bool separate_diag );
  bool solve(CNC_MatrixCRS<double> &A,
             const CNC_Vector<double> &b,
             CNC_Vector<double> &x,
             const unsigned int nb_iter_max,
             const double epsilon,
             Integer& nb_iteration,
             Real* residual_norm) ;
  void cublas_get_error(cublasStatus_t);
 public:
  void *gpu_r ;
  void *gpu_d ;
  void *gpu_h ;
  void *gpu_Ad ;
  void *gpu_diag_inv;
  void *gpu_b;
  void *gpu_x;
  void *gpu_temp;
  void *gpu_temp0;
  void *gpu_temp1;
  void *gpu_res0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //_IALEPH_CUDA_H_
