// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseFunctionProvider.h                                     (C) 2000-2023 */
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

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service fournissant des fonctions
 * utilisateur pour le JDD.
 *
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT ICaseFunctionProvider
{
 public:
	
  virtual ~ICaseFunctionProvider() = default; //!< Libère les ressources

 public:

  /*!
   * \brief Enregistre dans \a cm les fonctions fournies par ce service.
   */
  virtual void registerCaseFunctions(ICaseMng* cm) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service fournissant des fonctions
 * utilisateur pour le JDD.
 *
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT ICaseFunctionDotNetProvider
{
 public:

  virtual ~ICaseFunctionDotNetProvider() = default; //!< Libère les ressources

 public:

  /*!
   * \brief Enregistre dans \a cm les fonctions d'une classe '.Net'.
   */
  virtual void registerCaseFunctions(ICaseMng* cm,
                                     const String& assembly_name,
                                     const String& class_name) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

