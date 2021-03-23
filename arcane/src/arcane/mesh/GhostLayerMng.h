// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostLayerMng.h                                             (C) 2000-2013 */
/*                                                                           */
/* Gestionnaire de couche fantômes d'un maillage.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_GHOSTLAYERMNG_H
#define ARCANE_MESH_GHOSTLAYERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/IGhostLayerMng.h"
#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Gestionnaire de couche fantômes d'un maillage.
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

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
