// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMesh.h                                                     (C) 2000-2024 */
/*                                                                           */
/* Interface of a mesh.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESH_H
#define ARCANE_CORE_IMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/IMeshBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;
class MeshItemInternalList;
class IParticleExchanger;
class XmlNode;
class IMeshUtilities;
class IMeshModifier;
class IMeshMng;
class Properties;
class IMeshPartitionConstraintMng;
class IExtraGhostCellsBuilder;
class IUserData;
class IUserDataList;
class IGhostLayerMng;
class IMeshChecker;
class IMeshCompactMng;
class MeshPartInfo;
class IItemFamilyNetwork;
class MeshHandle;
class IVariableMng;
class ItemTypeMng;
class IMeshUniqueIdMng;
class MeshEventArgs;
enum class eMeshEventType;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//INFO: The complete documentation is in Mesh.dox
class IMesh
: public IMeshBase
{
 public:

  virtual ~IMesh() = default; //<! Releases resources

 public:

  virtual void build() = 0;

  //! Name of the factory used to create the mesh
  virtual String factoryName() const = 0;

  //! Internal array of mesh elements of type \a type
  virtual ItemInternalList itemsInternal(eItemKind) = 0;

  //! Node coordinates
  virtual SharedVariableNodeReal3 sharedNodesCoordinates() = 0;

  //! Check for the validity of internal mesh structures (internal)
  virtual void checkValidMesh() = 0;

  /*!
   * \brief Mesh validity check.
   *
   * This is a global check across all subdomains.
   *
   * It notably checks that connectivity is consistent between
   * subdomains.
   *
   * The check can be quite CPU intensive.
   * This method is collective.
   */
  virtual void checkValidMeshFull() = 0;

  /*!
   * \brief Synchronizes all mesh groups and variables.
   *
   * This operation is collective
   */
  virtual void synchronizeGroupsAndVariables() = 0;

 public:

  /*! \brief True if the mesh is allocated.
   *
   * A mesh is allocated as soon as an entity has been added, by allocateCells(),
   *  or reloadMesh()
   */
  virtual bool isAllocated() = 0;

  /*!
   * \brief Counter indicating the time of the last mesh modification.
   *
   * This counter increases with every call to endUpdate(). It is 0 upon
   * initialization. It allows, for example, checking if the mesh topology
   * has changed between two parts of the code.
   */
  virtual Int64 timestamp() = 0;

 public:

  //! Associated subdomain
  ARCANE_DEPRECATED_LONG_TERM("Y2020: Do not use this method. Try to get 'ISubDomain' from another way")
  virtual ISubDomain* subDomain() = 0;

 public:

  //! Parallelism manager
  virtual IParallelMng* parallelMng() = 0;

 public:

  //! Connectivity descriptor
  /*! This object allows reading/modifying connectivity */
  virtual VariableScalarInteger connectivity() = 0;

  //! AMR
  //! Group of all active cells
  virtual CellGroup allActiveCells() = 0;

  //! Group of all active cells specific to the domain
  virtual CellGroup ownActiveCells() = 0;

  //! Group of all cells of level \p level
  virtual CellGroup allLevelCells(const Integer& level) = 0;

  //! Group of all cells specific to the domain of level \p level
  virtual CellGroup ownLevelCells(const Integer& level) = 0;

  //! Group of all active faces
  virtual FaceGroup allActiveFaces() = 0;

  //! Group of all active faces specific to the domain.
  virtual FaceGroup ownActiveFaces() = 0;

  //! Group of all active faces
  virtual FaceGroup innerActiveFaces() = 0;

  //! Group of all active faces on the boundary.
  virtual FaceGroup outerActiveFaces() = 0;

 public:

  //! List of groups
  virtual ItemGroupCollection groups() = 0;

  //! Returns the group with name \a name or a null group if none exists.
  virtual ItemGroup findGroup(const String& name) = 0;

  //! Destroys all groups of all families.
  virtual void destroyGroups() = 0;

 public:

  virtual MeshItemInternalList* meshItemInternalList() = 0;

 public:

  virtual void updateGhostLayers(bool remove_old_ghost) = 0;

  /*!
   * \internal
   * \deprecated Use IMesh::cellFamily()->policyMng()->createSerializer() instead.
   */
  ARCANE_DEPRECATED_240 virtual void serializeCells(ISerializer* buffer, Int32ConstArrayView cells_local_id) = 0;

  //! Prepares the instance for dumping
  virtual void prepareForDump() = 0;

  //! Initializes variables with values from the configuration file (internal)
  virtual void initializeVariables(const XmlNode& init_node) = 0;

  /*!
   * \brief Sets the mesh check level.
   *
   * 0 - tests disabled
   * 1 - partial tests, after endUpdate()
   * 2 - full tests, after endUpdate()
   */
  virtual void setCheckLevel(Integer level) = 0;

  //! Current check level
  virtual Integer checkLevel() const = 0;

  //! Indicates if the mesh is dynamic (can evolve)
  virtual bool isDynamic() const = 0;

  //!
  virtual bool isAmrActivated() const = 0;

 public:

  //! \name Management of semi-conforming interfaces
  //@{
  //! Determines the semi-conforming interfaces
  virtual void computeTiedInterfaces(const XmlNode& mesh_node) = 0;

  //! True if semi-conforming interfaces exist in the mesh
  virtual bool hasTiedInterface() = 0;

  //! List of semi-conforming interfaces
  virtual TiedInterfaceCollection tiedInterfaces() = 0;
  //@}

  //! Manager of partitioning constraints associated with this mesh.
  virtual IMeshPartitionConstraintMng* partitionConstraintMng() = 0;

 public:

  //! Associated utility functions interface
  virtual IMeshUtilities* utilities() = 0;

  //! Properties associated with this mesh
  virtual Properties* properties() = 0;

 public:

  //! Associated modifier interface
  virtual IMeshModifier* modifier() = 0;

 public:

  /*!
   * \brief Node coordinates.
   *
   * Returns a native array (not shared like SharedVariable) of coordinates.
   * This call is only valid on a primary mesh (not a sub-mesh).
   */
  virtual VariableNodeReal3& nodesCoordinates() = 0;

  //@{ @name Sub-mesh interface
  /*!
   * \brief Defines the parent mesh and group.
   *
   *  Must be set on the mesh being constructed _before_ the build() phase
   */
  virtual void defineParentForBuild(IMesh* mesh, ItemGroup group) = 0;

  /*!
   * \brief Access to the parent mesh.
   *
   * Returns \a nullptr if the mesh does not have a parent mesh.
   */
  virtual IMesh* parentMesh() const = 0;

  /*!
   * \brief Parent group.
   *
   * Returns the null group if the mesh has no parent.
   */
  virtual ItemGroup parentGroup() const = 0;

  //! Adds a sub-mesh to the parent mesh
  virtual void addChildMesh(IMesh* sub_mesh) = 0;

  //! List of sub-meshes of the current mesh
  virtual MeshCollection childMeshes() const = 0;
  //@}

 public:

  /*!
   * \brief Indicates if the instance is a primary mesh.
   *
   * To be a primary mesh, the instance must
   * be convertible to an IPrimaryMesh
   * and not be a sub-mesh, meaning it
   * does not have a parent mesh (parentMesh()==nullptr).
   */
  virtual bool isPrimaryMesh() const = 0;

  /*!
   * \brief Returns the instance in the form of an IPrimaryMesh.
   *
   * Throws a BadCastException if the instance
   * is not of type IPrimaryMesh and if isPrimaryMesh() is false.
   */
  virtual IPrimaryMesh* toPrimaryMesh() = 0;

 public:

  //! Associated user data manager
  virtual IUserDataList* userDataList() = 0;

  //! Associated user data manager
  virtual const IUserDataList* userDataList() const = 0;

 public:

  //! Associated ghost layer manager
  virtual IGhostLayerMng* ghostLayerMng() const = 0;

  //! Unique ID numbering manager
  virtual IMeshUniqueIdMng* meshUniqueIdMng() const = 0;

  //! Checker interface.
  virtual IMeshChecker* checker() const = 0;

  //! Mesh part information
  virtual const MeshPartInfo& meshPartInfo() const = 0;

  //! check if the network itemFamily dependencies is activated
  virtual bool useMeshItemFamilyDependencies() const = 0;

  //! Family network interface (connected families)
  virtual IItemFamilyNetwork* itemFamilyNetwork() = 0;

  //! Interface of the indexed incremental connectivity manager.
  virtual IIndexedIncrementalItemConnectivityMng* indexedConnectivityMng() = 0;

  //! Mesh characteristics
  virtual const MeshKind meshKind() const = 0;

 public:

  //! Observable for an event
  virtual EventObservable<const MeshEventArgs&>& eventObservable(eMeshEventType type) = 0;

 public:

  //! \internal
  virtual IMeshCompactMng* _compactMng() = 0;

  /*!
   * \internal
   * \brief Connectivity usage policy
   */
  virtual InternalConnectivityPolicy _connectivityPolicy() const = 0;

 public:

  //! Associated mesh manager
  virtual IMeshMng* meshMng() const = 0;

  //! Associated variable manager
  virtual IVariableMng* variableMng() const = 0;

  //! Associated entity type manager
  virtual ItemTypeMng* itemTypeMng() const = 0;

 public:

  /*!
   * \brief Recalculates synchronization information.
   *
   * This operation is collective.
   *
   * Normally this is done automatically by %Arcane when it is
   * necessary. However, it can happen following certain internal modifications
   * that the information for synchronization needs to be manually updated.
   */
  virtual void computeSynchronizeInfos() = 0;

 public:

  //! Internal Arcane API
  virtual IMeshInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
