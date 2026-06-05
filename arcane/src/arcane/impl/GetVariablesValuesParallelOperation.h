// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GetVariablesValuesParallelOperation.h                       (C) 2000-2025 */
/*                                                                           */
/* Operations to access variable values from another subdomain.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_GETVARIABLESVALUESPARALLELOPERATION_H
#define ARCANE_IMPL_GETVARIABLESVALUESPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IGetVariablesValuesParallelOperation.h"

#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Operations to access variable values from another subdomain.
 */
class ARCANE_IMPL_EXPORT GetVariablesValuesParallelOperation
: public IGetVariablesValuesParallelOperation
{
 private:

  class Helper
  {
   public:

    SharedArray<Int64> m_unique_ids;
    SharedArray<Int32> m_indexes;
  };

 public:

  explicit GetVariablesValuesParallelOperation(IParallelMng* pm);

 public:

  IParallelMng* parallelMng() override;

 public:

  void getVariableValues(VariableItemReal& variable,
                         Int64ConstArrayView unique_ids,
                         RealArrayView values) override;

  void getVariableValues(VariableItemReal& variable,
                         Int64ConstArrayView unique_ids,
                         Int32ConstArrayView sub_domain_ids,
                         RealArrayView values) override;

 private:

  IParallelMng* m_parallel_mng = nullptr;

 private:

  template <class Type>
  void _getVariableValues(ItemVariableScalarRefT<Type>& variable,
                          Int64ConstArrayView unique_ids,
                          ArrayView<Type> values);

  template <class Type>
  void _getVariableValuesSequential(ItemVariableScalarRefT<Type>& variable,
                                    Int64ConstArrayView unique_ids,
                                    ArrayView<Type> values);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
