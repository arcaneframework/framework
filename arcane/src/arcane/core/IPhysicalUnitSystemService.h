// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPhysicalUnitSystemService.h                                (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service gérant un système d'unité.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPHYSICALUNITSYSTEMSERVICE_H
#define ARCANE_CORE_IPHYSICALUNITSYSTEMSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un service gérant un système d'unité.
 */
class ARCANE_CORE_EXPORT IPhysicalUnitSystemService
{
 public:

  virtual ~IPhysicalUnitSystemService() = default; //!< Libère les ressources.

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
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

