// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtraGhostParticlesBuilder.h                                (C) 2000-2024 */
/*                                                                           */
/* Construction des mailles fantômes supplémentaires.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_EXTRAGHOSTPARTICLESBUILDER_H
#define ARCANE_MESH_EXTRAGHOSTPARTICLESBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IExtraGhostParticlesBuilder;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;
class ParticleFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des mailles fantômes supplémentaires.
 */
class ExtraGhostParticlesBuilder
: public TraceAccessor
{
 public:

  explicit ExtraGhostParticlesBuilder(DynamicMesh* mesh);
  
 public:

  void computeExtraGhostParticles();
  void addExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder);
  void removeExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder);
  bool hasBuilder() const;

 private:
  
  DynamicMesh* m_mesh = nullptr;
  UniqueArray<IExtraGhostParticlesBuilder*> m_builders;

 private:

  void _computeForFamily(ParticleFamily* particle_family);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
