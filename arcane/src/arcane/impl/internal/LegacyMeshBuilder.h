// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LegacyMeshBuilder.h                                         (C) 2000-2020 */
/*                                                                           */
/* Construction du maillage via la méthode "historique".                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_LEGACYMESHBUILDER_H
#define ARCANE_IMPL_INTERNAL_LEGACYMESHBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/XmlNode.h"
#include "arcane/MeshHandle.h"
#include "arcane/IInitialPartitioner.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IPrimaryMesh;
class IInitialPartitioner;
class IMeshReader;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction du maillage via la méthode "historique".
 *
 * Ce mécanisme utilise l'élément '<mesh>' ou '<maillage>' du jeu de
 * donnée pour lire les informations et créer le maillage initial.
 */
class ARCANE_IMPL_EXPORT LegacyMeshBuilder
: public TraceAccessor
{
  struct MeshBuildInfo
  {
   public:
    MeshBuildInfo() : m_dir_name("."), m_mesh(nullptr) {}
   public:
    XmlNode m_xml_node;
    String m_orig_file_name;
    String m_file_name;
    String m_dir_name;
    MeshHandle m_mesh_handle;
    IPrimaryMesh* m_mesh;
  };

 public:

  LegacyMeshBuilder(ISubDomain* sd,MeshHandle default_mesh_handle);

 public:

  void readCaseMeshes();
  void createDefaultMesh();
  void readMeshes();
  void allocateMeshes();
  void initializeMeshVariablesFromCaseFile();

 private:
  ISubDomain* m_sub_domain;
  MeshHandle m_default_mesh_handle;
  //TODO rendre privé
 public:
  UniqueArray<MeshBuildInfo> m_meshes_build_info; //!< Infos pour construire un maillage
  ScopedPtrT<IInitialPartitioner> m_initial_partitioner; //!< Partitionneur initial
  String m_internal_partitioner_name;
  bool m_use_internal_mesh_partitioner = false; //!< \a true si partitionne le maillage en interne
  bool m_use_partitioner_tester = false; //!< basic partitioner for metis

 private:

  void _readMesh(ConstArrayView<Ref<IMeshReader>> mesh_readers,const MeshBuildInfo& mbi);
  void _createMeshesHandle();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
