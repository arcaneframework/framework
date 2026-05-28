// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCompactMng.h                                            (C) 2000-2016 */
/*                                                                           */
/* Mesh compacting manager for families of a mesh.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHCOMPACTMNG_H
#define ARCANE_MESH_MESHCOMPACTMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/core/IMeshCompactMng.h"
#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshCompacter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Manager for compacting families of a mesh.
 */
class ARCANE_MESH_EXPORT MeshCompactMng
: public TraceAccessor
, public IMeshCompactMng
{
 public:

  MeshCompactMng(IMesh* mesh);
  ~MeshCompactMng();

 public:

  IMesh* mesh() const override;
  IMeshCompacter* beginCompact() override;
  IMeshCompacter* beginCompact(IItemFamily* family) override;
  void endCompact() override;
  IMeshCompacter* compacter() override { return m_compacter; }

 private:

  IMesh* m_mesh;
  IMeshCompacter* m_compacter;

 private:

  MeshCompacter* _setCompacter(MeshCompacter* c);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
