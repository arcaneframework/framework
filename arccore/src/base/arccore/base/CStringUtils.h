// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CStringUtils.h                                              (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires sur les chaînes de caractères.                      */
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
 * \brief Fonctions utilitaires sur les chaînes de caractères.
 */
namespace CStringUtils
{
  //! Retourne \e true si \a s1 et \a s2 sont identiques, \e false sinon
  ARCCORE_BASE_EXPORT bool isEqual(const char* s1,const char* s2);

  //! Retourne \e true si \a s1 est inférieur (ordre alphabétique) à \a s2 , \e false sinon
  ARCCORE_BASE_EXPORT bool isLess(const char* s1,const char* s2);

  //! Retourne la longueur de la chaîne \a s (sur 32 bits)
  ARCCORE_BASE_EXPORT Integer len(const char* s);

  //! Retourne la longueur de la chaîne \a s
  ARCCORE_BASE_EXPORT Int64 largeLength(const char* s);

  // Copie les \a n premiers caractères de \a from dans \a to. \retval
  ARCCORE_BASE_EXPORT char* copyn(char* to,const char* from,Int64 n);

  //! Copie \a from dans \a to
  ARCCORE_BASE_EXPORT char* copy(char* to,const char* from);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

