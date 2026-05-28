// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostLayerMng.h                                             (C) 2000-2013 */
/*                                                                           */
/* Mesh ghost layer manager.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_GHOSTLAYERMNG_H
#define ARCANE_MESH_GHOSTLAYERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/core/IGhostLayerMng.h"
#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Mesh ghost layer manager.
 */
class GhostLayerMng
: public TraceAccessor
, public IGhostLayerMng
{
 public:

  explicit GhostLayerMng(ITraceMng* tm);

 public:

  void setNbGhostLayer(Integer n) override;
  Integer nbGhostLayer() const override;

  void setBuilderVersion(Integer n) override;
  Integer builderVersion() const override;

 private:

  Integer m_nb_ghost_layer;
  Integer m_builder_version;

 private:

  void _initBuilderVersion();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/

#endif
