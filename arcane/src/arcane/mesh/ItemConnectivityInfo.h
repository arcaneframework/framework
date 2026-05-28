// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivityInfo.h                                      (C) 2000-2022 */
/*                                                                           */
/* Information on connectivity by entity type.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMCONNECTIVITYINFO_H
#define ARCANE_MESH_ITEMCONNECTIVITYINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IItemConnectivityInfo.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemInternalConnectivityList;
class IParallelMng;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Information on connectivity by entity type.
 */
class ARCANE_MESH_EXPORT ItemConnectivityInfo
: public IItemConnectivityInfo
{
  enum ICI_Type
  {
    ICI_Node = 0,
    ICI_Edge,
    ICI_Face,
    ICI_Cell
  };
  static const int NB_ICI = 4;

 public:

  ItemConnectivityInfo();

 public:

  Integer maxNodePerItem() const override { return m_infos[ICI_Node]; }
  Integer maxEdgePerItem() const override { return m_infos[ICI_Edge]; }
  Integer maxFacePerItem() const override { return m_infos[ICI_Face]; }
  Integer maxCellPerItem() const override { return m_infos[ICI_Cell]; }

 public:

  void fill(ItemInternalConnectivityList* clist);
  void reduce(IParallelMng* pm);

 private:

  Integer m_infos[NB_ICI];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
