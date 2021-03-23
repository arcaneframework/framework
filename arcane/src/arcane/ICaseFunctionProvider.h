// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseFunctionProvider.h                                     (C) 2000-2011 */
/*                                                                           */
/* Interface d'un service fournissant des fonctions utilisateur pour le JDD. */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ICASEFUNCTIONPROVIDER_H
#define ARCANE_ICASEFUNCTIONPROVIDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service fournissant des fonctions
 * utilisateur pour le JDD.
 *
 * \ingroup CaseOption
 *
 */
class ICaseFunctionProvider
{
 public:


 public:
	
  virtual ~ICaseFunctionProvider(){} //!< Libère les ressources

 public:

  /*!
   * \brief Enregistre dans \a cm les fonctions fournies par ce service.
   */
  virtual void registerCaseFunctions(ICaseMng* cm) =0;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

