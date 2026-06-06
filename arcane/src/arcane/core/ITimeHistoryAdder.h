// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryAdder.h                                         (C) 2000-2024 */
/*                                                                           */
/* Class interface allowing the addition of a value history linked to        */
/* a mesh.                                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ITIMEHISTORYMNGADDER_H
#define ARCANE_ITIMEHISTORYMNGADDER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/core/ITimeHistoryMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class interface allowing the addition of one or more values
 * to a value history.
 */
class ITimeHistoryAdder
{
 public:

  virtual ~ITimeHistoryAdder() = default; //!< Frees resources

 public:

  /*!
   * \brief Method allowing the addition of a value to a history.
   *
   * \param thpi The parameters of the value.
   * \param value The value to add.
   */
  virtual void addValue(const TimeHistoryAddValueArg& thp, Real value) = 0;

  /*!
   * \brief Method allowing the addition of a value to a history.
   *
   * \param thpi The parameters of the value.
   * \param value The value to add.
   */
  virtual void addValue(const TimeHistoryAddValueArg& thp, Int32 value) = 0;

  /*!
   * \brief Method allowing the addition of a value to a history.
   *
   * \param thpi The parameters of the value.
   * \param value The value to add.
   */
  virtual void addValue(const TimeHistoryAddValueArg& thp, Int64 value) = 0;

  /*!
   * \brief Method allowing the addition of values to a history.
   *
   * \param thpi The parameters of the values.
   * \param value The values to add.
   */
  virtual void addValue(const TimeHistoryAddValueArg& thp, RealConstArrayView values) = 0;

  /*!
   * \brief Method allowing the addition of values to a history.
   *
   * \param thpi The parameters of the values.
   * \param value The values to add.
   */
  virtual void addValue(const TimeHistoryAddValueArg& thp, Int32ConstArrayView values) = 0;

  /*!
   * \brief Method allowing the addition of values to a history.
   *
   * \param thpi The parameters of the values.
   * \param value The values to add.
   */
  virtual void addValue(const TimeHistoryAddValueArg& thp, Int64ConstArrayView values) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
