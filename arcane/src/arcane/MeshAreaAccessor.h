// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshAreaAccessor.h                                          (C) 2000-2004 */
/*                                                                           */
/* Accès aux informations d'une zone de maillage.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHAREAACCESSOR_H
#define ARCANE_MESHAREAACCESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshArea;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Accès aux informations d'une zone de maillage.
 */
class ARCANE_CORE_EXPORT MeshAreaAccessor
{
 public:

  MeshAreaAccessor(IMeshArea* mesh_area);
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
  
  IMeshArea* m_mesh_area;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

