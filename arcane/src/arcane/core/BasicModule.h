// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicModule.h                                               (C) 2000-2025 */
/*                                                                           */
/* Module basique avec informations de maillage.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_BASICMODULE_H
#define ARCANE_CORE_BASICMODULE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/AbstractModule.h"
#include "arcane/core/MeshAccessor.h"
#include "arcane/core/CommonVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Module
 *
 * \brief Module basique.
 *
 * Classe de base d'un module permettant de récupérer aisément les informations
 * de maillages (IMesh) et les variables standards (CommonVariables).
 */
class ARCANE_CORE_EXPORT BasicModule
: public AbstractModule
, public MeshAccessor
, public CommonVariables
{
 protected:

  //! Constructeur à partir d'un \a ModuleBuildInfo
  explicit BasicModule(const ModuleBuildInfo&);

 public:

  //! Destructeur
  ~BasicModule() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
