// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtraGhostParticlesBuilder.h                                (C) 2011-2020 */
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

  ExtraGhostParticlesBuilder(DynamicMesh* mesh);
  
 public:

  void addExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder)
  {
    m_builders.add(builder);
  }
  
  ArrayView<IExtraGhostParticlesBuilder*> extraGhostParticlesBuilders()
  {
    return m_builders;
  }
  
  void computeExtraGhostParticles();
  
 private:
  
  DynamicMesh* m_mesh;
  UniqueArray<IExtraGhostParticlesBuilder*> m_builders;

  void _computeForFamily(ParticleFamily* particle_family);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
