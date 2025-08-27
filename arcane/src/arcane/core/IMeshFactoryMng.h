// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshFactoryMng.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire de fabriques de maillages.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHFACTORYMNG_H
#define ARCANE_CORE_IMESHFACTORYMNG_H
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
 * \brief Interface du gestionnaire de fabriques de maillages.
 */
class ARCANE_CORE_EXPORT IMeshFactoryMng
{
 public:

  //! Libère les ressources.
  virtual ~IMeshFactoryMng() = default;

 public:

  //! Gestionnaire de maillage associé
  virtual IMeshMng* meshMng() const =0;

  /*!
   * \brief Créé un maillage ou un sous-maillage.
   *
   * Le maillage créé est automatiquement ajouté au meshMng() associé.
   */
  virtual IPrimaryMesh* createMesh(const MeshBuildInfo& build_info) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
