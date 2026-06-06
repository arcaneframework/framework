// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseFunctionProvider.h                                     (C) 2000-2023 */
/*                                                                           */
/* Interface of a service providing user functions for the dataset.          */
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
 * \brief Interface of a service providing user functions
 * for the JDD.
 *
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT ICaseFunctionProvider
{
 public:
	
  virtual ~ICaseFunctionProvider() = default; //!< Releases resources

 public:

  /*!
   * \brief Registers the functions provided by this service in \a cm.
   */
  virtual void registerCaseFunctions(ICaseMng* cm) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a service providing user functions
 * for the JDD.
 *
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT ICaseFunctionDotNetProvider
{
 public:

  virtual ~ICaseFunctionDotNetProvider() = default; //!< Releases resources

 public:

  /*!
   * \brief Registers the functions of a '.Net' class in \a cm.
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
