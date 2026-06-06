// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshBuildInfo.h                                             (C) 2000-2023 */
/*                                                                           */
/* Information for building a mesh.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHBUILDINFO_H
#define ARCANE_MESHBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/MeshKind.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Parameters necessary for building a mesh.
 *
 * Only the name (name()) is essential. The other parameters are
 * optional depending on the desired creation type. If they are not
 * defined and useful, a default value will be used.
 *
 * There are two possibilities for creation:
 *
 * - creation of a new mesh. In this case, it is necessary to specify
 *   the associated \a IParallelMng via addParallelMng().
 * - creation of a sub-mesh of an existing mesh. In this case, it is necessary
 *   to position the group via addItemGroup(). The
 *   associated IParallelMng must be that of the parent mesh. The
 *   created sub-mesh will include the meshes of this group.
 */
class ARCANE_CORE_EXPORT MeshBuildInfo
{
 public:

  /*!
   * \brief Constructs a default mesh with the name \a name.
   */
  explicit MeshBuildInfo(const String& name);

 public:

  //! Sets the factory name to create this mesh
  MeshBuildInfo& addFactoryName(const String& factory_name);
  //! Sets the parallelism manager to create the mesh
  MeshBuildInfo& addParallelMng(Ref<IParallelMng> pm);
  //! Sets the mesh group for a sub-mesh
  MeshBuildInfo& addParentGroup(const ItemGroup& parent_group);
  /*!
  * \brief Indicates whether the generator needs to call a partitioner.
  *
  * This is the case, for example, if the reader can only generate sequential meshes.
  */
  MeshBuildInfo& addNeedPartitioning(bool v);

  //! Sets the mesh characteristics
  MeshBuildInfo& addMeshKind(const MeshKind& v);

  //! Name of the new mesh
  const String& name() const { return m_name; }
  //! Factory name to create the mesh (via IMeshFactory)
  const String& factoryName() const { return m_factory_name; }
  //! Parallelism manager in the case of a new mesh.
  Ref<IParallelMng> parallelMngRef() const { return m_parallel_mng; }
  //! Parent group in the case of a sub-mesh, null otherwise.
  const ItemGroup& parentGroup() const { return m_parent_group; }
  //! Indicates if the reader/generator requires partitioning
  bool isNeedPartitioning() const { return m_is_need_partitioning; }
  //! Mesh characteristics
  const MeshKind meshKind() const { return m_mesh_kind; }

 private:

  String m_name;
  String m_factory_name;
  Ref<IParallelMng> m_parallel_mng;
  ItemGroup m_parent_group;
  bool m_is_need_partitioning = false;
  MeshKind m_mesh_kind;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
