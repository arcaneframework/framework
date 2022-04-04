// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CStringUtils.h                                              (C) 2000-2015 */
/*                                                                           */
/* Fonctions utilitaires sur les chaînes de caractères.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_CSTRINGUTILS_H
#define ARCANE_UTILS_CSTRINGUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions utilitaires sur les chaînes de caractères.
 */
namespace CStringUtils
{
  /*!
   * \brief Converti la chaîne \a str en un réel.
   * Si \a is_ok n'est pas nul, il vaut \a true en retour si la conversion
   * est correcte, false sinon.
   * \return la valeur de str convertie en réel ou 0. en cas d'erreur.
   */
  ARCANE_UTILS_EXPORT Real toReal(const char* str,bool* is_ok=0);
  /*!
   * \brief Converti la chaîne \a str en un entier non signé.
   * Si \a is_ok n'est pas nul, il vaut \a true en retour si la conversion
   * est correcte, false sinon.
   * \return la valeur de str convertie en entier non signé ou 0 en cas d'erreur.
   */
  ARCANE_UTILS_EXPORT Integer toInteger(const char* str,bool* is_ok=0);

  /*!
   * \brief Converti la chaîne \a str en un entier
   * Si \a is_ok n'est pas nul, il vaut \a true en retour si la conversion
   * est correcte, false sinon.
   * \return la valeur de str convertie en entier ou 0 en cas d'erreur.
   */
  ARCANE_UTILS_EXPORT int toInt(const char* str,bool* is_ok=0);
 
  //! Retourne \e true si \a s1 et \a s2 sont identiques, \e false sinon
  ARCANE_UTILS_EXPORT bool isEqual(const char* s1,const char* s2);

  //! Retourne \e true si \a s1 est inférieur (ordre alphabétique) à \a s2 , \e false sinon
  ARCANE_UTILS_EXPORT bool isLess(const char* s1,const char* s2);

  //! Retourne la longueur de la chaîne \a s
  ARCANE_UTILS_EXPORT Integer len(const char* s);

  /*! \brief Copie les \a n premiers caractères de \a from dans \a to.
   * \retval to */
  ARCANE_UTILS_EXPORT char* copyn(char* to,const char* from,Integer n);

  //! Copie \a from dans \a to
  ARCANE_UTILS_EXPORT char* copy(char* to,const char* from);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

