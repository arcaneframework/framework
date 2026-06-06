// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableUtilities.h                                        (C) 2000-2025 */
/*                                                                           */
/* Interface providing utility functions on variables.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEUTILITIES_H
#define ARCANE_CORE_IVARIABLEUTILITIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface providing utility functions on variables.
 */
class ARCANE_CORE_EXPORT IVariableUtilities
{
 public:

  virtual ~IVariableUtilities() = default; //!< Frees resources.

 public:

  //! Associated variable manager
  virtual IVariableMng* variableMng() const = 0;

  /*!
   * \brief Displays dependency information for a variable.
   *
   * Displays on the stream \a ostr the information about the variables
   * that depend on \a var. If \a is_recursive is true, this
   * method is also called for these variables.
   */
  virtual void dumpDependencies(IVariable* var, std::ostream& ostr, bool is_recursive) = 0;

  /*!
   * \brief Displays dependency information for all variables.
   *
   * Displays on the stream \a ostr the information of all
   * used variables.
   */
  virtual void dumpAllVariableDependencies(std::ostream& ostr, bool is_recursive) = 0;

  /*!
   * \brief Filters common variables between multiple ranks.
   *
   * This method allows filtering the variables in \a input_variables
   * that are present on all ranks of \a pm. It returns
   * the list sorted alphabetically of variables common to all
   * ranks.
   *
   * If \a dump_no_common is true, it displays (via ITraceMng::info()) the list
   * of variables that are not common on all ranks.
   */
  virtual VariableCollection filterCommonVariables(IParallelMng* pm,
                                                   VariableCollection input_variables,
                                                   bool dump_not_common) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
