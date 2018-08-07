/*---------------------------------------------------------------------------*/
/* CStringUtils.cc                                             (C) 2000-2018 */
/*                                                                           */
/* Fonctions utilitaires sur les chaînes de caractères.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/CStringUtils.h"

#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
  return (std::strcmp(s1,s2) < 0);
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

extern "C++" ARCCORE_BASE_EXPORT void
initializeStringConverter()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
