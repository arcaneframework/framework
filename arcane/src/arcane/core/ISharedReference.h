﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISharedReference.h                                          (C) 2000-2006 */
/*                                                                           */
/* Interface de la classe compteur de référence.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ISHAREDREFERENCE_H
#define ARCANE_ISHAREDREFERENCE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ptr.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Interface d'un compteur de référence.
 *
 Le compteur de référence permet à une instance classe de connaître le nombre
 de références sur elle. Lorsque ce nombre arrive à zéro, cela signifie
 que l'instance n'est plus utilisée. Ce système est utilisé principalement
 pour libérer automatiquement la mémoire lorsque le nombre de références
 tombe à zéro.
 
 Cette classe s'utilise par l'intermédiaire de classes comme AutoRefT qui
 permettent d'incrémenter ou de décrémenter automatiquement le compteur
 de l'objet sur lesquelles elles pointent.

 \since 0.2.9
 \author Gilles Grospellier
 \date 06/10/2000
 */
class ARCANE_CORE_EXPORT ISharedReference
{
 public:

  //! Libère les ressources
  virtual ~ISharedReference(){}

 public:
	
  //! Incrémente le compteur de référence
  virtual void addRef() =0;

  //! Décrémente le compteur de référence
  virtual void removeRef() =0;

  //! Retourne la valeur du compteur de référence
  virtual Int32 refCount() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

