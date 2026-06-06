// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableFilter.h                                           (C) 2000-2025 */
/*                                                                           */
/* Functor of a filter applicable to variables.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEFILTER_H
#define ARCANE_CORE_IVARIABLEFILTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Variable
 * \brief Functor of a filter applicable to variables.
 *
 * This functor is used when iterating over variables. The applyFilter() method
 * is called for each variable and indicates whether the tested variable should
 * or should not be filtered.
 */
class IVariableFilter
{
 public:

  virtual ~IVariableFilter() = default; //!< Releases resources

  /*!
   * \brief Applies the filter to the variable \a var.
   * \retval true if the variable meets the filter conditions
   * \retval false otherwise.
   */
  virtual bool applyFilter(IVariable& var) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
