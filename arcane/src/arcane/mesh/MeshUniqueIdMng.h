// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUniqueIdMng.h                                           (C) 2000-2021 */
/*                                                                           */
/* Gestionnaire de numérotation des uniqueId() d'un maillage.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHUNIQUEIDMNG_H
#define ARCANE_MESH_MESHUNIQUEIDMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/IMeshUniqueIdMng.h"
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

  void setFaceBuilderVersion(Integer n) override;
  Integer faceBuilderVersion() const override { return m_face_builder_version; }

  void setEdgeBuilderVersion(Integer n) override;
  Integer edgeBuilderVersion() const override { return m_edge_builder_version; }

 private:

  Integer m_face_builder_version;
  Integer m_edge_builder_version;

 private:

  void _initFaceVersion();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
