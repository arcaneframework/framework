// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CStringUtils.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Utility functions for strings.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/CStringUtils.h"

#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Returns true if s1 and s2 are identical, false otherwise
bool CStringUtils::
isEqual(const char* s1, const char* s2)
{
  if (!s1 && !s2)
    return true;
  if (!s1 || !s2)
    return false;
  while (*s1 == *s2) {
    if (*s1 == '\0')
      return true;
    ++s1;
    ++s2;
  }
  return false;
}

//! Returns true if s1 is less than (alphabetical order) s2, false otherwise
bool CStringUtils::
isLess(const char* s1, const char* s2)
{
  if (!s1 && !s2)
    return false;
  if (!s1 || !s2)
    return false;
  return (std::strcmp(s1, s2) < 0);
}

//! Returns the length of the string s
Integer CStringUtils::
len(const char* s)
{
  if (!s)
    return 0;
  //return (Integer)::strlen(s);
  Integer len = 0;
  while (s[len] != '\0')
    ++len;
  return len;
}

//! Returns the length of the string s
Int64 CStringUtils::
largeLength(const char* s)
{
  if (!s)
    return 0;
  return std::strlen(s);
}

char* CStringUtils::
copyn(char* to, const char* from, Int64 n)
{
  return ::strncpy(to, from, n);
}

char* CStringUtils::
copy(char* to, const char* from)
{
  return ::strcpy(to, from);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
initializeStringConverter()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
