// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterListWithCaseOption.h                               (C) 2000-2025 */
/*                                                                           */
/* Parameter list with support for dataset options.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_PARAMETERLISTWITHCASEOPTION_H
#define ARCANE_UTILS_INTERNAL_PARAMETERLISTWITHCASEOPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/internal/ParameterCaseOption.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ParameterList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Parameter list with information to override dataset options.
 */
class ARCANE_UTILS_EXPORT ParameterListWithCaseOption
{
  class Impl;

 public:

  //! Constructs a dictionary
  ParameterListWithCaseOption();
  //! Constructs a dictionary
  ParameterListWithCaseOption(const ParameterListWithCaseOption& rhs);
  ~ParameterListWithCaseOption(); //!< Releases resources

 public:

  /*!
   * \brief Retrieves the parameter with name \a param_name.
   *
   * Returns a null string if no parameter with this name exists.
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
   * In case (1) or (3), the argument value is added to the
   * already present occurrences. In case (2), the argument value
   * replaces all already present occurrences. In case (4), the occurrence
   * having the value \a B is deleted if it was present and nothing happens if it was absent.
   *
   * \retval false if a parameter could be parsed
   * \retval true otherwise.
   */
  bool addParameterLine(const String& line);

  /*!
   * \brief Method to retrieve an object of type ParameterCaseOption.
   *
   * This object can be destroyed after use.
   *
   * \param language The language in which the dataset is written.
   * \return An object of type ParameterCaseOption.
   */
  ParameterCaseOption getParameterCaseOption(const String& language) const;

  //! Adds the parameters from \a parameters to the instance's parameters
  void addParameters(const ParameterList& parameters);

 private:

  Impl* m_p = nullptr; //!< Implementation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
