// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CStringUtils.h                                              (C) 2000-2025 */
/*                                                                           */
/* Utility functions for strings.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_CSTRINGUTILS_H
#define ARCCORE_BASE_CSTRINGUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Utility functions for strings.
 */
namespace CStringUtils
{
  //! Returns \e true if \a s1 and \a s2 are identical, \e false otherwise
  ARCCORE_BASE_EXPORT bool isEqual(const char* s1,const char* s2);

  //! Returns \e true if \a s1 is less than (alphabetical order) \a s2 , \e false otherwise
  ARCCORE_BASE_EXPORT bool isLess(const char* s1,const char* s2);

  //! Returns the length of the string \a s (in 32 bits)
  ARCCORE_BASE_EXPORT Integer len(const char* s);

  //! Returns the length of the string \a s
  ARCCORE_BASE_EXPORT Int64 largeLength(const char* s);

  // Copies the first \a n characters of \a from into \a to. \retval
  ARCCORE_BASE_EXPORT char* copyn(char* to,const char* from,Int64 n);

  //! Copies \a from into \a to
  ARCCORE_BASE_EXPORT char* copy(char* to,const char* from);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
