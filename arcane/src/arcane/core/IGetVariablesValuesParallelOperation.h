// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGetVariablesValuesParallelOperation.h                      (C) 2000-2025 */
/*                                                                           */
/* Operations to access variable values from another subdomain.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IGETVARIABLESVALUESPARALLELOPERATION_H
#define ARCANE_CORE_IGETVARIABLESVALUESPARALLELOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/VariableTypedef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Operations to access variable values from another subdomain.
 * \todo use serialization + templates to support all variable types.
 */
class ARCANE_CORE_EXPORT IGetVariablesValuesParallelOperation
{
 public:

  virtual ~IGetVariablesValuesParallelOperation() = default;

 public:

  virtual IParallelMng* parallelMng() = 0;

 public:

  /*!
   * \brief Retrieves the values of a variable on remote entities
   *
   * This operation allows retrieving the values of the variable
   * \a variable on entities that are not located in this subdomain.
   * The array \a unique_ids contains the <b>unique</b> number of the entities
   * whose value we wish to retrieve. These values will be stored
   * in \a values.
   *
   * This method generally requires a lot of communication
   * because it is necessary to search which subdomain the entities
   * belong to based on their uniqueId(). If the subdomain is known, it
   * is better to use the overloaded method with this parameter.
   *
   * \a unique_ids and \a values must have the same number of elements.
   *
   * This operation is collective and blocking.
   */
  virtual void getVariableValues(VariableItemReal& variable,
                                 Int64ConstArrayView unique_ids,
                                 RealArrayView values) = 0;
  /*!
   * \brief Retrieves the values of a variable on remote entities
   *
   * This operation allows retrieving the values of the variable
   * \a variable on entities that are not located in this subdomain.
   * The array \a unique_ids contains the <b>unique</b> number of the entities
   * whose value we wish to retrieve, and \a sub_domain_ids the subdomain
   * in which the entities are located. These values will be stored in \a values.
   *
   * \a unique_ids, \a sub_domain_ids, and \a values must have
   * the same number of elements.
   *
   * This operation is collective and blocking.
   */
  virtual void getVariableValues(VariableItemReal& variable,
                                 Int64ConstArrayView unique_ids,
                                 Int32ConstArrayView sub_domain_ids,
                                 RealArrayView values) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
