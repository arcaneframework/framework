// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephVector.h                                                    (C) 2010 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_VECTOR_H
#define ALEPH_VECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/AlephGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IAlephVector;
class AlephTopology;
class AlephKernel;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur d'un système linéaire.
 */
class ARCANE_ALEPH_EXPORT AlephVector : public TraceAccessor
{
 public:
  AlephVector(AlephKernel*);
  ~AlephVector();

 public:
  void create(void);
  void create_really(void);
  void update(void) { throw NotImplementedException(A_FUNCINFO); }
  void reSetLocalComponents(AlephVector*);
  void setLocalComponents(Integer num_values,
                          ConstArrayView<int> glob_indices,
                          ConstArrayView<double> values);
  void setLocalComponents(ConstArrayView<double> values);
  void getLocalComponents(Integer vector_size,
                          ConstArrayView<int> global_indice,
                          ArrayView<double> vector_values);
  void getLocalComponents(Array<double>& values);
  void startFilling(void);
  void assemble(void);
  void assemble_waitAndFill(void);
  void reassemble(void);
  void reassemble_waitAndFill(void);
  void copy(AlephVector*) { throw NotImplementedException(A_FUNCINFO); }
  void writeToFile(const String);
  IAlephVector* implementation(void) { return m_implementation; }

 private:
  AlephKernel* m_kernel;
  Integer m_index;
  ArrayView<Integer> m_ranks;
  bool m_participating_in_solver;
  IAlephVector* m_implementation;

 private:
  // Buffers utilisés dans le cas où nous sommes le solveur
  UniqueArray<Int32> m_aleph_vector_buffer_idxs;
  UniqueArray<Real> m_aleph_vector_buffer_vals;

 private:
  UniqueArray<Int32> m_aleph_vector_buffer_idx;
  UniqueArray<Real> m_aleph_vector_buffer_val;

 private:
  UniqueArray<Parallel::Request> m_parallel_requests;
  UniqueArray<Parallel::Request> m_parallel_reassemble_requests;

 public:
  Integer m_bkp_num_values;
  UniqueArray<int> m_bkp_indexs;
  UniqueArray<double> m_bkp_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
