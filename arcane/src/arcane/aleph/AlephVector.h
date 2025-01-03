// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephVector.h                                               (C) 2000-2024 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ALEPH_ALEPHVECTOR_H
#define ARCANE_ALEPH_ALEPHVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/AlephGlobal.h"

#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur d'un système linéaire.
 */
class ARCANE_ALEPH_EXPORT AlephVector
: public TraceAccessor
{
 public:

  explicit AlephVector(AlephKernel*);
  ~AlephVector();

 public:

  void create();
  void create_really();
  void update() { throw NotImplementedException(A_FUNCINFO); }
  void reSetLocalComponents(AlephVector*);
  void setLocalComponents(Integer num_values,
                          ConstArrayView<AlephInt> glob_indices,
                          ConstArrayView<double> values);
  void setLocalComponents(ConstArrayView<double> values);
  void getLocalComponents(Integer vector_size,
                          ConstArrayView<AlephInt> global_indice,
                          ArrayView<double> vector_values);
  void getLocalComponents(Array<double>& values);
  void startFilling();
  void assemble();
  void assemble_waitAndFill();
  void reassemble();
  void reassemble_waitAndFill();
  void copy(AlephVector*) { throw NotImplementedException(A_FUNCINFO); }
  void writeToFile(const String);
  IAlephVector* implementation(void) { return m_implementation; }

 private:

  AlephKernel* m_kernel = nullptr;
  Integer m_index = -1;
  ArrayView<Integer> m_ranks;
  bool m_participating_in_solver = false;
  IAlephVector* m_implementation = nullptr;

 private:

  // Buffers utilisés dans le cas où nous sommes le solveur
  UniqueArray<AlephInt> m_aleph_vector_buffer_idxs;
  UniqueArray<Real> m_aleph_vector_buffer_vals;
  UniqueArray<AlephInt> m_aleph_vector_buffer_idx;
  UniqueArray<Real> m_aleph_vector_buffer_val;
  UniqueArray<Parallel::Request> m_parallel_requests;
  UniqueArray<Parallel::Request> m_parallel_reassemble_requests;

 public:

  Integer m_bkp_num_values;
  UniqueArray<AlephInt> m_bkp_indexs;
  UniqueArray<double> m_bkp_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
