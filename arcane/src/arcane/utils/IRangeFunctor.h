// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRangeFunctor.h                                             (C) 2000-2010 */
/*                                                                           */
/* Interface d'un fonctor sur un interval d'itération.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IRANGEFUNCTOR_H
#define ARCANE_UTILS_IRANGEFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un fonctor sur un interval d'itération.
 * \ingroup Core
 */
class ARCANE_UTILS_EXPORT IRangeFunctor
{
 public:
	
  //! Libère les ressources
  virtual ~IRangeFunctor(){}

 public:

  /*!
   * \brief Exécute la méthode associée.
   * \param begin indice du début de l'itération.
   * \param size nombre d'éléments à itérer.
   */
  virtual void executeFunctor(Integer begin,Integer size) =0;  
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

