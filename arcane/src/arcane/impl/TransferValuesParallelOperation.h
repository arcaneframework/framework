// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TransferValuesParallelOperation.h                           (C) 2000-2020 */
/*                                                                           */
/* Transfert de valeurs sur différents processeurs.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_TRANSFERVALUESPARALLELOPERATION_H
#define ARCANE_IMPL_TRANSFERVALUESPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ITransferValuesParallelOperation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérations pour transférer des valeurs vers d'autres sous-domaine.
 */
class ARCANE_IMPL_EXPORT TransferValuesParallelOperation
: public ITransferValuesParallelOperation
{
 public:
  TransferValuesParallelOperation(IParallelMng* pm);
 public:
  //! Destructeur
  virtual ~TransferValuesParallelOperation();
 public:
  virtual IParallelMng* parallelMng();
 public:
  virtual void setTransferRanks(Int32ConstArrayView ranks);
  virtual void addArray(Int32ConstArrayView send_values,SharedArray<Int32> recv_value);
  virtual void addArray(Int64ConstArrayView send_values,SharedArray<Int64> recv_values);
  virtual void addArray(RealConstArrayView send_values,SharedArray<Real> recv_values);
  virtual void transferValues();
 private:
  IParallelMng* m_parallel_mng;
  Int32ConstArrayView m_ranks;
  UniqueArray< Int32ConstArrayView > m_send32_values;
  UniqueArray< Int64ConstArrayView > m_send64_values;
  UniqueArray< RealConstArrayView > m_send_real_values;
  UniqueArray< SharedArray<Int32> > m_recv32_values;
  UniqueArray< SharedArray<Int64> > m_recv64_values;
  UniqueArray< SharedArray<Real> > m_recv_real_values;
 private:
  template<typename U>
  void _putArray(ISerializer* s,
                 Span<const Integer> z_indexes,
                 UniqueArray< ConstArrayView<U> >& arrays,
                 Array<U>& tmp_values);
  template<typename U> void
  _getArray(ISerializer* s, Integer nb,
            UniqueArray< SharedArray<U> >& arrays,
            Array<U>& tmp_values);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
