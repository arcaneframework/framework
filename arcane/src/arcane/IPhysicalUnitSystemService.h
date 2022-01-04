// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPhysicalUnitSystemService.h                                (C) 2000-2010 */
/*                                                                           */
/* Interface d'un service gérant un système d'unité.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPHYSICALUNITSYSTEMSERVICE_H
#define ARCANE_IPHYSICALUNITSYSTEMSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IPhysicalUnitSystem;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un service gérant un système d'unité.
 */
class ARCANE_CORE_EXPORT IPhysicalUnitSystemService
{
 public:

  virtual ~IPhysicalUnitSystemService() {} //!< Libère les ressources.

 public:

  virtual void build() =0;

 public:

  /*!
   * \brief Crée un système d'unité pour le Système International SI.
   */
  virtual IPhysicalUnitSystem* createStandardUnitSystem() =0;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

