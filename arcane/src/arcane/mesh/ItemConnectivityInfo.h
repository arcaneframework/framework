// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivityInfo.h                                      (C) 2000-2013 */
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

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
  virtual ~ItemConnectivityInfo() {} //<! Libère les ressources

 public:

 public:

  virtual Integer maxNodePerItem() const
  { return m_infos[ICI_Node]; }
  virtual Integer maxEdgePerItem() const
  { return m_infos[ICI_Edge]; }
  virtual Integer maxFacePerItem() const
  { return m_infos[ICI_Face]; }
  virtual Integer maxCellPerItem() const
  { return m_infos[ICI_Cell]; }
  virtual Integer maxNodeInItemTypeInfo() const
  { return m_infos[ICI_NodeItemTypeInfo]; }
  virtual Integer maxEdgeInItemTypeInfo() const
  { return m_infos[ICI_EdgeItemTypeInfo]; }
  virtual Integer maxFaceInItemTypeInfo() const
  { return m_infos[ICI_FaceItemTypeInfo]; }

 public:

  void fill(ItemSharedInfoList* isl);
  void reduce(IParallelMng* pm);

 private:

  Integer m_infos[NB_ICI];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
