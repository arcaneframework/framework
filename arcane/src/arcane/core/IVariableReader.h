// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableReader.h                                           (C) 2000-2025 */
/*                                                                           */
/* Reading of variables for initialization and during calculation.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEREADER_H
#define ARCANE_CORE_IVARIABLEREADER_H
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
 * \internal
 * \brief Brief reading of variables during calculation.
 */
class IVariableReader
{
 public:

  virtual ~IVariableReader() = default;

 public:

  //! Sets the path of the directory containing the data
  virtual void setBaseDirectoryName(const String& path) = 0;
  //! Sets the name of the file containing the data.
  virtual void setBaseFileName(const String& filename) = 0;
  /*!
   * \brief Initializes the reader.
   *
   * \a is_start is true if we are at the start of the calculation.
   */
  virtual void initialize(bool is_start) = 0;
  /*!
   * \brief Sets the list of variables that we wish to reread.
   * This call must happen before initialize().
   */
  virtual void setVariables(VariableCollection vars) = 0;
  //! Updates the variables for the time \a wanted_time
  virtual void updateVariables(Real wanted_time) = 0;
  /*!
   * \brief Time interval of values for the variable \a var.
   * The data for the variable \a var exists for the times
   * included between \a a.x and \a a.y with \a a having the value
   * returned.
   *
   * This call is valid only after calling initialize().
   */
  virtual Real2 timeInterval(IVariable* var) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
