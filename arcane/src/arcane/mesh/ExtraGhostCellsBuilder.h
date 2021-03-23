// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtraGhostCellsBuilder.h                                    (C) 2011-2011 */
/*                                                                           */
/* Construction des mailles fantômes supplémentaires.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_EXTRAGHOSTCELLSBUILDER_H
#define ARCANE_MESH_EXTRAGHOSTCELLSBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IExtraGhostCellsBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des mailles fantômes supplémentaires.
 */
class ExtraGhostCellsBuilder
  : public TraceAccessor
{
public:

  ExtraGhostCellsBuilder(DynamicMesh* mesh);
  
  ~ExtraGhostCellsBuilder() {}

public:

  void addExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder) {
    m_builders.add(builder);
  }
  
  ArrayView<IExtraGhostCellsBuilder*> extraGhostCellsBuilders() {
    return m_builders;
  }
  
  void computeExtraGhostCells();
  
private:
  
  DynamicMesh* m_mesh;

  UniqueArray<IExtraGhostCellsBuilder*> m_builders;  
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
