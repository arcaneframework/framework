// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemsOwnerBuilder.h                                         (C) 2000-2025 */
/*                                                                           */
/* Classe pour calculer les propriétaires des entités.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITESMOWNERBUILDER_H
#define ARCANE_MESH_ITESMOWNERBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Arcane
{
class IMesh;
}
namespace Arcane::mesh
{
class ItemsOwnerBuilderImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe générique pour calculer les propriétaires des entités.
 *
 * Pour toutes les méthodes, on suppose que les propriétaires des mailles
 * sont correctement valides et sont synchronisés.
 */
class ARCANE_MESH_EXPORT ItemsOwnerBuilder
{
  class Impl;

 public:

  explicit ItemsOwnerBuilder(IMesh* mesh);
  ~ItemsOwnerBuilder();

 public:

  void computeFacesOwner();
  void computeNodesOwner();

 private:

  std::unique_ptr<ItemsOwnerBuilderImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
