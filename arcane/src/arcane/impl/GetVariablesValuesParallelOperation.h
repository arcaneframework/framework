// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GetVariablesValuesParallelOperation.h                       (C) 2000-2021 */
/*                                                                           */
/* Opérations pour accéder aux valeurs de variables d'un autre sous-domaine. */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_GETVARIABLESVALUESPARALLELOPERATION_H
#define ARCANE_IMPL_GETVARIABLESVALUESPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IGetVariablesValuesParallelOperation.h"

#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérations pour accéder aux valeurs de variables d'un autre sous-domaine.
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
  GetVariablesValuesParallelOperation(IParallelMng* pm);
  virtual ~GetVariablesValuesParallelOperation();

 public:

  virtual IParallelMng* parallelMng();

 public:

  virtual void getVariableValues(VariableItemReal& variable,
                                 Int64ConstArrayView unique_ids,
                                 RealArrayView values);

  virtual void getVariableValues(VariableItemReal& variable,
                                 Int64ConstArrayView unique_ids,
                                 Int32ConstArrayView sub_domain_ids,
                                 RealArrayView values);
 private:

  IParallelMng* m_parallel_mng;

 private:

  void _deleteMessages(Array<ISerializeMessage*>& messages);

  template<class Type>
  void _getVariableValues(ItemVariableScalarRefT<Type>& variable,
                          Int64ConstArrayView unique_ids,
                          ArrayView<Type> values);

  template<class Type>
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
