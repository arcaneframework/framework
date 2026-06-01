// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CStringUtils.h                                              (C) 2000-2015 */
/*                                                                           */
/* Utility functions for character strings.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_CSTRINGUTILS_H
#define ARCANE_UTILS_CSTRINGUTILS_H
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
 * \brief Utility functions for character strings.
 */
namespace CStringUtils
{
  /*!
   * \brief Converts the string \a str to a real number.
   * If \a is_ok is not null, it is set to \a true if the conversion
   * is correct, false otherwise.
   * \return the value of str converted to a real number or 0 in case of error.
   */
  ARCANE_UTILS_EXPORT Real toReal(const char* str, bool* is_ok = 0);
  /*!
   * \brief Converts the string \a str to an unsigned integer.
   * If \a is_ok is not null, it is set to \a true if the conversion
   * is correct, false otherwise.
   * \return the value of str converted to an unsigned integer or 0 in case of error.
   */
  ARCANE_UTILS_EXPORT Integer toInteger(const char* str, bool* is_ok = 0);

  /*!
   * \brief Converts the string \a str to an integer
   * If \a is_ok is not null, it is set to \a true if the conversion
   * is correct, false otherwise.
   * \return the value of str converted to an integer or 0 in case of error.
   */
  ARCANE_UTILS_EXPORT int toInt(const char* str, bool* is_ok = 0);

  //! Returns \e true if \a s1 and \a s2 are identical, \e false otherwise
  ARCANE_UTILS_EXPORT bool isEqual(const char* s1, const char* s2);

  //! Returns \e true if \a s1 is less than (alphabetical order) \a s2 , \e false otherwise
  ARCANE_UTILS_EXPORT bool isLess(const char* s1, const char* s2);

  //! Returns the length of the string \a s
  ARCANE_UTILS_EXPORT Integer len(const char* s);

  /*! \brief Copies the first \a n characters of \a from into \a to.
   * \retval to */
  ARCANE_UTILS_EXPORT char* copyn(char* to, const char* from, Integer n);

  //! Copies \a from into \a to
  ARCANE_UTILS_EXPORT char* copy(char* to, const char* from);
} // namespace CStringUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
