// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshAreaAccessor.h                                          (C) 2000-2025 */
/*                                                                           */
/* Accès aux informations d'une zone de maillage.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHAREAACCESSOR_H
#define ARCANE_CORE_MESHAREAACCESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Accès aux informations d'une zone de maillage.
 */
class ARCANE_CORE_EXPORT MeshAreaAccessor
{
 public:

  explicit MeshAreaAccessor(IMeshArea* mesh_area);
  ~MeshAreaAccessor();

 public:

  //! Zone de maillage accédée par cette accessor
  IMeshArea* meshArea();

  //! Positionne à \a mesh_area la zone de maillage accédée par cette accessor
  void setMeshArea(IMeshArea* mesh_area);

 public:

  //! Nombre de noeuds du maillage
  Integer nbNode();

  //! Nombre de mailles du maillage
  Integer nbCell();

 public:

  //! Groupe de tous les noeuds de la zone
  NodeGroup allNodes();

  //! Groupe de toutes les mailles de la zone
  CellGroup allCells();

  //! Groupe de tous les noeuds propres de la zone
  NodeGroup ownNodes();

  //! Groupe de toutes les mailles propres de la zone
  CellGroup ownCells();

 private:

  IMeshArea* m_mesh_area = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
