// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShapeMng.cc                                             (C) 2000-2014 */
/*                                                                           */
/* Classe gérant les GeomShape d'un maillage.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMesh.h"

#include "arcane/geometric/GeomShapeMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GeomShapeMng::
GeomShapeMng(IMesh* mesh,const String& cell_shape_name)
: m_name(cell_shape_name)
, m_cell_shape_nodes(VariableBuildInfo(mesh,cell_shape_name,IVariable::PNoDump))
, m_cell_shape_faces(VariableBuildInfo(mesh,cell_shape_name+"Face",IVariable::PNoDump))
, m_cell_shape_centers(VariableBuildInfo(mesh,cell_shape_name+"Center",IVariable::PNoDump))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GeomShapeMng::
GeomShapeMng(IMesh* mesh)
: m_name("GenericElement")
, m_cell_shape_nodes(VariableBuildInfo(mesh,"GenericElement",IVariable::PNoDump))
, m_cell_shape_faces(VariableBuildInfo(mesh,"GenericElementFace",IVariable::PNoDump))
, m_cell_shape_centers(VariableBuildInfo(mesh,"GenericElementCenter",IVariable::PNoDump))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GeomShapeMng::
GeomShapeMng(const GeomShapeMng& rhs)
: m_name(rhs.m_name)
, m_cell_shape_nodes(rhs.m_cell_shape_nodes)
, m_cell_shape_faces(rhs.m_cell_shape_faces)
, m_cell_shape_centers(rhs.m_cell_shape_centers)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeMng::
initialize()
{
  IMesh* mesh = m_cell_shape_nodes.variable()->mesh();
  //TODO: il faut utiliser le globalConnectivity() de IItemFamily
  // mais pour l'instant celui-ci n'est pas correctement calculé
  // lors de l'init.
  if (mesh->dimension()==2){
    // En 2D, on n'a pas des mailles contenant plus de noeuds que les quads
    m_cell_shape_nodes.resize(4);
    //TODO: Verifier si on a besoin de cela.
    m_cell_shape_faces.resize(4);
  }
  else{
    m_cell_shape_nodes.resize(ItemStaticInfo::MAX_CELL_NODE);
    m_cell_shape_faces.resize(ItemStaticInfo::MAX_CELL_FACE);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

