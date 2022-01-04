// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IUserDataList.h                                             (C) 2000-2018 */
/*                                                                           */
/* Interface d'une liste qui gère des données utilisateurs.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IUSERDATALIST_H
#define ARCANE_UTILS_IUSERDATALIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IUserData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une liste qui gère des données utilisateurs.
 * \ingroup Core
 */
class ARCANE_UTILS_EXPORT IUserDataList
{
 public:
	
  //! Libère les ressources
  virtual ~IUserDataList(){}

 public:

  /*!
   * \brief Positionne le user-data associé au nom \a name.
   *
   * Aucune donnée ne doit déjà être associée à \a name, sinon une
   * exception est levée.
   */  
  virtual void setData(const String& name,IUserData* ud) =0;
  /*!
   * \brief Donnée associée à \a name.
   *
   * Une exception est levée si \a allow_null vaut \a false et qu'aucune
   * donnée n'est associée à \a name. Si \a allow_null est \a vrai et
   * qu'aucune donnée n'est associée, retourne un pointeur nul.
   */
  virtual IUserData* data(const String& name,bool allow_null=false) const =0;
  /*!
   * \brief Supprime la donnée associèe au nom \a name.
   *
   * Une exception est levée si \a allow_null vaut \a false et qu'aucune
   * donnée n'est associée à \a name.
   */
  virtual void removeData(const String& name,bool allow_null=false) =0;

  /*!
   * \brief Supprime toutes les données utilisateurs.
   *
   * Cela revient à appeler removeData() pour toutes les données utilisateurs.
   */  
  virtual void clear() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

