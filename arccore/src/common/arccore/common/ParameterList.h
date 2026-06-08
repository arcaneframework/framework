// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterList.h                                             (C) 2000-2025 */
/*                                                                           */
/* Parameter list.                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_PARAMETERLIST_H
#define ARCCORE_COMMON_PARAMETERLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ParameterCaseOption;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Parameter list.
 *
 * A parameter list is similar to a set (key,value) but
 * a key may potentially be present multiple times (a bit like the
 * std::multi_map class).
 */
class ARCCORE_COMMON_EXPORT ParameterList
{
 private:

  friend class ParameterListWithCaseOption;
  class Impl; //!< Implementation

 public:

  //! Constructs a dictionary
  ParameterList();
  //! Constructs a dictionary
  ParameterList(const ParameterList& rhs);
  ~ParameterList(); //!< Frees resources

 public:

  /*!
   * \brief Retrieves the parameter with name \a param_name.
   *
   * Returns a null string if there is no parameter with this name.
   *
   * If the parameter is present multiple times, only the last
   * value is returned.
   */
  String getParameterOrNull(const String& param_name) const;

  /*!
   * \brief Parses the line \a line.
   *
   * The line must have one of the following forms, with \a A the
   * parameter and \a B the value:
   *
   * 1. A=B,
   * 2. A:=B
   * 3. A+=B,
   * 4. A-=B
   *
   * In case (1) or (3), the argument's value is added to the
   * already present occurrences. In case (2), the argument's value
   * replaces all already present occurrences. In
   * case (4), the occurrence having the value \a B is deleted if it
   * was present and nothing happens if it was absent.
   *
   * \retval false if a parameter could be parsed
   * \retval true otherwise.
   */
  bool addParameterLine(const String& line);

  /*!
   * \brief Retrieves the list of parameters and their values.
   *
   * Returns in \a param_names the list of parameter names and
   * in \a values the associated value.
   */
  void fillParameters(StringList& param_names, StringList& values) const;

 private:

  Impl* m_p = nullptr; //!< Implementation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
