// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CStringUtils.cc                                             (C) 2000-2015 */
/*                                                                           */
/* Fonctions utilitaires sur les chaînes de caractères.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/StdHeader.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/CStringUtils.h"
#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real CStringUtils::
toReal(const char* str,bool* is_ok)
{
  Real v = 0.;
  bool is_bad = builtInGetValue(v,str);
  if (is_ok)
    *is_ok = !is_bad;
  return v;
}

Integer
CStringUtils::
toInteger(const char* str,bool* is_ok)
{
  Integer v = 0;
  bool is_bad = builtInGetValue(v,str);
  if (is_ok)
    *is_ok = !is_bad;
  return v;
}

int
CStringUtils::
toInt(const char* str,bool* is_ok)
{
  int v = 0;
  bool is_bad = builtInGetValue(v,str);
  if (is_ok)
    *is_ok = !is_bad;
  return v;
}

//! Retourne \e true si \a s1 et \a s2 sont identiques, \e false sinon
bool CStringUtils::
isEqual(const char* s1,const char* s2)
{
  if (!s1 && !s2)
    return true;
  if (!s1 || !s2)
    return false;
  while (*s1==*s2){
    if (*s1=='\0')
      return true;
    ++s1; ++s2;
  }
  return false;
}

//! Retourne \e true si \a s1 est inférieur (ordre alphabétique) à \a s2 , \e false sinon
bool CStringUtils::
isLess(const char* s1,const char* s2)
{
  if (!s1 && !s2)
    return false;
  if (!s1 || !s2)
    return false;
  return (::strcmp(s1,s2) < 0);
}

//! Retourne la longueur de la chaîne \a s
Integer CStringUtils::
len(const char* s)
{
  if (!s) return 0;
  //return (Integer)::strlen(s);
  Integer len = 0;
  while(s[len]!='\0')
    ++len;
  return len;
}

char* CStringUtils::
copyn(char* to,const char* from,Integer n)
{
  return ::strncpy(to,from,n);
}

char* CStringUtils::
copy(char* to,const char* from)
{
  return ::strcpy(to,from);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
initializeStringConverter()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
