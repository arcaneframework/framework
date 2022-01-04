// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IUserData.h                                                 (C) 2000-2012 */
/*                                                                           */
/* Interface pour une donnée utilisateur attachée à un autre objet.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IUSERDATA_H
#define ARCANE_UTILS_IUSERDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour une donnée utilisateur attachée à un autre objet.
 * \ingroup Core
 */
class ARCANE_UTILS_EXPORT IUserData
{
 public:
	
  //! Libère les ressources
  virtual ~IUserData(){}

 public:

  //! Méthode exécutée lorsque l'instance est attachée.
  virtual void notifyAttach() =0;

  //! Méthode exécutée lorsque l'instance est détachée.
  virtual void notifyDetach() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

