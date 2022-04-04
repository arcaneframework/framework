// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NameComparer.h                                              (C) 2000-2006 */
/*                                                                           */
/* Classe utilitaire pour la destruction des objets alloués par new.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NAMECOMPARER_H
#define ARCANE_UTILS_NAMECOMPARER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 \brief Classe utilitaire pour comparer le nom d'une instance.
  
  Compare l'attribut name() d'un objet avec la valeur donnée dans
  le constructeur.

  \warning L'instance de cette classe ne fait pas de recopie de
  la chaîne donnée dans le constructeur. Elle doit donc rester
  valide tant que cette instance l'est.
*/
class NameComparer
{
 public:
  NameComparer(const String& s)
  : m_name(s) {}
 public:
  template<typename U> inline bool
  operator()(const U* ptr) const
    {
      return ptr->name() == m_name;
    }
 private:
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
