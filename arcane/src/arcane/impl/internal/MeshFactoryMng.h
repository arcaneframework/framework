// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshFactoryMng.h                                            (C) 2000-2020 */
/*                                                                           */
/* Gestionnaire de fabriques de maillages.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_MESHFACTORYMNG_H
#define ARCANE_IMPL_INTERNAL_MESHFACTORYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMeshFactoryMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class MeshMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT MeshFactoryMng
: public IMeshFactoryMng
{
 public:

  MeshFactoryMng(IApplication* app,MeshMng* mesh_mng);

 public:

  IMeshMng* meshMng() const override;
  IPrimaryMesh* createMesh(const MeshBuildInfo& build_info) override;

 private:

  IApplication* m_application;
  MeshMng* m_mesh_mng;

 private:

  IPrimaryMesh* _createMesh(const MeshBuildInfo& build_info);
  IPrimaryMesh* _createSubMesh(const MeshBuildInfo& build_info);
  void _checkValidBuildInfo(const MeshBuildInfo& build_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
