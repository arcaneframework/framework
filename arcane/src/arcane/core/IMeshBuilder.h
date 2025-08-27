// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshReader.h                                               (C) 2000-2020 */
/*                                                                           */
/* Interface d'un service de créatrion/lecture du maillage.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHBUILDER_H
#define ARCANE_IMESHBUILDER_H
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
 * \ingroup StandardService
 * \brief Interface d'un service de création/lecture du maillage.
 */
class ARCANE_CORE_EXPORT IMeshBuilder
{
 public:

  virtual ~IMeshBuilder() = default; //<! Libère les ressources

 public:

  /*!
   * \brief Remplit \a build_info avec les informations nécessaires pour
   * créer le maillage.
   *
   * Certaines valeurs peuvent être remplies par l'appelant mais l'instance
   * peut éventuellement les surcharger. En particulier, il est possible
   * de spécifier la fabrique de maillage à utiliser.
   */
  virtual void fillMeshBuildInfo(MeshBuildInfo& build_info) =0;

  //! Alloue les entités du maillage géré par ce service.
  virtual void allocateMeshItems(IPrimaryMesh* pm) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

