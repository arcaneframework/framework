// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IFunctorWithAddress.h                                       (C) 2000-2012 */
/*                                                                           */
/* Interface d'un fonctor.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IFUNCTOR_WITH_ADDRESS_H
#define ARCANE_UTILS_IFUNCTOR_WITH_ADDRESS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IFunctor.h"
#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un fonctor.
 * \ingroup Core
 */
class ARCANE_UTILS_EXPORT IFunctorWithAddress
: public IFunctor
{
 public:
	
  //! Libère les ressources
  virtual ~IFunctorWithAddress(){}

 public:

  /*!
   * \internal
   * \brief Retourne l'adresse de la méthode associé.
   * \warning Cette méthode ne doit être appelée que par HYODA
   * et n'est pas valide sur toutes les plate-formes.
   */
  virtual void* functorAddress() =0;  
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

