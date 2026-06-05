// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LegacyMeshBuilder.h                                         (C) 2000-2023 */
/*                                                                           */
/* Mesh construction via the "historical" method.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_LEGACYMESHBUILDER_H
#define ARCANE_IMPL_INTERNAL_LEGACYMESHBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/XmlNode.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/IInitialPartitioner.h"

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
 * \brief Mesh construction via the "historical" method.
 *
 * This mechanism uses the '<mesh>' or '<maillage>' element of the data set
 * to read the information and create the initial mesh.
 */
class ARCANE_IMPL_EXPORT LegacyMeshBuilder
: public TraceAccessor
{
  struct MeshBuildInfo
  {
   public:

    MeshBuildInfo()
    : m_dir_name(".")
    , m_mesh(nullptr)
    {}

   public:

    XmlNode m_xml_node;
    String m_orig_file_name;
    String m_file_name;
    String m_dir_name;
    MeshHandle m_mesh_handle;
    IPrimaryMesh* m_mesh;
  };

 public:

  LegacyMeshBuilder(ISubDomain* sd, MeshHandle default_mesh_handle);

 public:

  void readCaseMeshes();
  void createDefaultMesh();
  void readMeshes();
  void allocateMeshes();
  void initializeMeshVariablesFromCaseFile();

 private:

  ISubDomain* m_sub_domain;
  MeshHandle m_default_mesh_handle;
  //TODO make private
 public:

  UniqueArray<MeshBuildInfo> m_meshes_build_info; //!< Info to build a mesh
  ScopedPtrT<IInitialPartitioner> m_initial_partitioner; //!< Initial partitioner
  String m_internal_partitioner_name;
  bool m_use_internal_mesh_partitioner = false; //!< \a true if it partitions the mesh internally
  bool m_use_partitioner_tester = false; //!< basic partitioner for metis

 private:

  void _readMesh(ConstArrayView<Ref<IMeshReader>> mesh_readers, const MeshBuildInfo& mbi);
  void _createMeshesHandle();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
