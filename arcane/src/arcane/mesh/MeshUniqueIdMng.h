// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUniqueIdMng.h                                           (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de numérotation des uniqueId() d'un maillage.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHUNIQUEIDMNG_H
#define ARCANE_MESH_MESHUNIQUEIDMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/IMeshUniqueIdMng.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de numérotation des uniqueId() d'un maillage.
 */
class ARCANE_MESH_EXPORT MeshUniqueIdMng
: public TraceAccessor
, public IMeshUniqueIdMng
{
 public:

  explicit MeshUniqueIdMng(ITraceMng* tm);

 public:

  void setFaceBuilderVersion(Int32 n) override;
  Int32 faceBuilderVersion() const override { return m_face_builder_version; }

  void setEdgeBuilderVersion(Int32 n) override;
  Int32 edgeBuilderVersion() const override { return m_edge_builder_version; }

  void setUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId(bool v) override;
  bool isUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId() const override
  {
    return m_use_node_uid_to_generate_edge_and_face_uid;
  }

 private:

  Int32 m_face_builder_version = 1;
  Int32 m_edge_builder_version = 1;
  bool m_use_node_uid_to_generate_edge_and_face_uid = false;

 private:

  void _initFaceVersion();
  void _initEdgeVersion();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
