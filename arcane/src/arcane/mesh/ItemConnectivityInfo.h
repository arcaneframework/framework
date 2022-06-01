// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivityInfo.h                                      (C) 2000-2022 */
/*                                                                           */
/* Informations sur la connectivité par type d'entité.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMCONNECTIVITYINFO_H
#define ARCANE_MESH_ITEMCONNECTIVITYINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IItemConnectivityInfo.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemInternalConnectivityList;
class IParallelMng;
}

namespace Arcane::mesh
{
class ItemSharedInfoList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur la connectivité par type d'entité.
 */
class ARCANE_MESH_EXPORT ItemConnectivityInfo
: public IItemConnectivityInfo
{
  enum ICI_Type
  {
    ICI_Node = 0,
    ICI_Edge,
    ICI_Face,
    ICI_Cell,
    ICI_NodeItemTypeInfo,
    ICI_EdgeItemTypeInfo,
    ICI_FaceItemTypeInfo
  };
  static const int NB_ICI = 7;

 public:

  ItemConnectivityInfo();

 public:

  Integer maxNodePerItem() const override { return m_infos[ICI_Node]; }
  Integer maxEdgePerItem() const override { return m_infos[ICI_Edge]; }
  Integer maxFacePerItem() const override { return m_infos[ICI_Face]; }
  Integer maxCellPerItem() const override { return m_infos[ICI_Cell]; }

 public:

  void fill(ItemSharedInfoList* isl,ItemInternalConnectivityList* clist);
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
