// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableUtilsInternal.h                                     (C) 2000-2024 */
/*                                                                           */
/* Various utility functions for internal Arcane variables.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_VARIABLEUTILSINTERNAL_H
#define ARCANE_CORE_INTERNAL_VARIABLEUTILSINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT VariableUtilsInternal
{
 public:

  /*!
   * \brief Fills \a values with the variable's values.
   *
   * Only 1D variables of type \a DT_Real are convertible.
   *
   * \retval false if everything went well.
   * \retval true if nothing was done.
   */
  static bool fillFloat64Array(IVariable* v, ArrayView<double> values);

  /*!
   * \brief Copies the values \a values into the variable \a v.
   *
   * Only 1D variables of type \a DT_Real are convertible.
   *
   * \retval false if everything went well.
   * \retval true if nothing was done.
   */
  static bool setFromFloat64Array(IVariable* v, ConstArrayView<double> values);

  /*!
   * \brief Copies the values \a values into the variable \a v.
   *
   * Only numerical variables are convertible.
   *
   * \retval false if everything went well.
   * \retval true if nothing was done.
   */
  static bool setFromMemoryBuffer(IVariable* v, ConstMemoryView values);

  //! Returns the internal IData API associated with the variable \a v
  static IDataInternal* getDataInternal(IVariable* v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::VariableUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
