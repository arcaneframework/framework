// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EmptyMesh                                                   (C) 2000-2023 */
/*                                                                           */
/* Brief code description                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_EMPTYMESH_H
#define ARCANE_EMPTYMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IUserDataList.h"
#include "arcane/utils/IUserData.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/IMesh.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/VariableTypedef.h"
#include "arcane/IParallelMng.h"
#include "arcane/MeshItemInternalList.h"
#include "arcane/XmlNode.h"
#include "arcane/IMeshPartitionConstraintMng.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IMeshChecker.h"
#include "arcane/IMeshCompactMng.h"
#include "arcane/IMeshMng.h"
#include "arcane/IGhostLayerMng.h"
#include "arcane/Properties.h"
#include "arcane/MeshPartInfo.h"
#include "arcane/IItemFamilyNetwork.h"
#include "arcane/IItemFamily.h"
#include "arcane/IVariableMng.h"
#include "arcane/MeshVariableScalarRef.h"
#include "arcane/SharedVariable.h"
#include "arcane/VariableRefScalar.h"
#include "arcane/MeshHandle.h"
#include "arcane/IParticleExchanger.h"
#include "arcane/IExtraGhostCellsBuilder.h"
#include "arcane/core/MeshKind.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EmptyMesh
: public IPrimaryMesh
{

 public:

  ITraceMng* m_trace_mng;

  explicit EmptyMesh(ITraceMng* trace_mng)
  : m_trace_mng(trace_mng)
  {}
  ~EmptyMesh() override = default;

 private:

  void _error() const { m_trace_mng->fatal() << "Using EmptyMesh"; }

 public:

  MeshHandle handle() const override
  {
    _error();
    return MeshHandle{};
  }

  // IPrimaryMesh API
 public:

  VariableNodeReal3& nodesCoordinates() override
  {
    _error();
    auto var = new VariableNodeReal3{ nullptr };
    return *var;
  }
  void setDimension(Integer) override { _error(); }
  void reloadMesh() override { _error(); }
  void allocateCells(Integer, Int64ConstArrayView, bool) override { _error(); }
  void endAllocate() override { _error(); }
  void deallocate() override { _error(); }
  VariableItemInt32& itemsNewOwner(eItemKind) override
  {
    _error();
    auto var = new VariableItemInt32{ nullptr };
    return *var;
  };
  void exchangeItems() override { _error(); }
  void setOwnersFromCells() override { _error(); }
  void setMeshPartInfo(const MeshPartInfo&) override { _error(); }

  // IMesh API
 public:

  String name() const override
  {
    _error();
    return String{};
  }
  Integer nbNode() override
  {
    _error();
    return -1;
  }
  Integer nbEdge() override
  {
    _error();
    return -1;
  }
  Integer nbFace() override
  {
    _error();
    return -1;
  }
  Integer nbCell() override
  {
    _error();
    return -1;
  }
  Integer nbItem(eItemKind) override
  {
    _error();
    return -1;
  }
  ITraceMng* traceMng() override
  {
    _error();
    return nullptr;
  }
  Integer dimension() override
  {
    _error();
    return -1;
  }
  NodeGroup allNodes() override
  {
    _error();
    return NodeGroup{};
  }
  EdgeGroup allEdges() override
  {
    _error();
    return EdgeGroup{};
  }
  FaceGroup allFaces() override
  {
    _error();
    return FaceGroup{};
  }
  CellGroup allCells() override
  {
    _error();
    return CellGroup{};
  }
  NodeGroup ownNodes() override
  {
    _error();
    return NodeGroup{};
  }
  EdgeGroup ownEdges() override
  {
    _error();
    return EdgeGroup{};
  }
  FaceGroup ownFaces() override
  {
    _error();
    return FaceGroup{};
  }
  CellGroup ownCells() override
  {
    _error();
    return CellGroup{};
  }
  FaceGroup outerFaces() override
  {
    _error();
    return FaceGroup{};
  }

 public:

  IItemFamily* createItemFamily(eItemKind, const String&) override
  {
    _error();
    return nullptr;
  }
  IItemFamily* findItemFamily(eItemKind, const String&, bool, bool) override
  {
    _error();
    return nullptr;
  }
  IItemFamily* findItemFamily(const String&, bool) override
  {
    _error();
    return nullptr;
  }
  IItemFamilyModifier* findItemFamilyModifier(eItemKind, const String&) override
  {
    _error();
    return nullptr;
  }
  IItemFamily* itemFamily(eItemKind) override
  {
    _error();
    return nullptr;
  }
  IItemFamily* nodeFamily() override
  {
    _error();
    return nullptr;
  }
  IItemFamily* edgeFamily() override
  {
    _error();
    return nullptr;
  }
  IItemFamily* faceFamily() override
  {
    _error();
    return nullptr;
  }
  IItemFamily* cellFamily() override
  {
    _error();
    return nullptr;
  }
  IItemFamilyCollection itemFamilies() override
  {
    _error();
    return IItemFamilyCollection{};
  }

 public:

  void build() override { _error(); }
  String factoryName() const override
  {
    _error();
    return String{};
  }
  ItemInternalList itemsInternal(eItemKind) override
  {
    _error();
    return ItemInternalList{};
  }
  SharedVariableNodeReal3 sharedNodesCoordinates() override
  {
    _error();
    return SharedVariableNodeReal3{};
  }
  void checkValidMesh() override { _error(); }
  void checkValidMeshFull() override { _error(); }
  void synchronizeGroupsAndVariables() override { _error(); }

 public:

  bool isAllocated() override
  {
    _error();
    return false;
  }
  Int64 timestamp() override
  {
    _error();
    return -1;
  }

 public:

  ISubDomain* subDomain() override
  {
    _error();
    return nullptr;
  }

 public:

  IParallelMng* parallelMng() override
  {
    _error();
    return nullptr;
  }

 public:

  VariableScalarInteger connectivity() override
  {
    _error();
    return VariableScalarInteger{ nullptr };
  }

  CellGroup allActiveCells() override
  {
    _error();
    return CellGroup{};
  }
  CellGroup ownActiveCells() override
  {
    _error();
    return CellGroup{};
  }
  CellGroup allLevelCells(const Integer&) override
  {
    _error();
    return CellGroup{};
  }
  CellGroup ownLevelCells(const Integer&) override
  {
    _error();
    return CellGroup{};
  }
  FaceGroup allActiveFaces() override
  {
    _error();
    return FaceGroup{};
  }
  FaceGroup ownActiveFaces() override
  {
    _error();
    return FaceGroup{};
  }
  FaceGroup innerActiveFaces() override
  {
    _error();
    return FaceGroup{};
  }
  FaceGroup outerActiveFaces() override
  {
    _error();
    return FaceGroup{};
  }

 public:

  ItemGroupCollection groups() override
  {
    _error();
    return ItemGroupCollection{};
  }
  ItemGroup findGroup(const String&) override
  {
    _error();
    return ItemGroup{};
  }
  void destroyGroups() override { _error(); }

 public:

  MeshItemInternalList* meshItemInternalList() override
  {
    _error();
    return nullptr;
  }

 public:

  void updateGhostLayers(bool) override { _error(); }
  void serializeCells(ISerializer*, Int32ConstArrayView) override { _error(); }
  void prepareForDump() override { _error(); }
  void initializeVariables(const XmlNode&) override { _error(); }
  void setCheckLevel(Integer) override { _error(); }
  Integer checkLevel() const override
  {
    _error();
    return -1;
  }
  bool isDynamic() const override
  {
    _error();
    return false;
  }
  bool isAmrActivated() const override
  {
    _error();
    return false;
  }

  eMeshAMRKind amrType() const override
  {
    _error();
    return eMeshAMRKind::None;
  }

 public:

  void computeTiedInterfaces(const XmlNode&) override { _error(); }
  bool hasTiedInterface() override
  {
    _error();
    return false;
  }
  TiedInterfaceCollection tiedInterfaces() override
  {
    _error();
    return TiedInterfaceCollection{};
  }
  IMeshPartitionConstraintMng* partitionConstraintMng() override
  {
    _error();
    return nullptr;
  }

 public:

  IMeshUtilities* utilities() override
  {
    _error();
    return nullptr;
  }
  Properties* properties() override
  {
    _error();
    return nullptr;
  }

 public:

  IMeshModifier* modifier() override
  {
    _error();
    return nullptr;
  }

 public:

  void defineParentForBuild(IMesh*, ItemGroup) override { _error(); }
  IMesh* parentMesh() const override
  {
    _error();
    return nullptr;
  }
  ItemGroup parentGroup() const override
  {
    _error();
    return ItemGroup{};
  }
  void addChildMesh(IMesh*) override { _error(); }
  MeshCollection childMeshes() const override
  {
    _error();
    return MeshCollection{};
  }

 public:

  bool isPrimaryMesh() const override
  {
    _error();
    return false;
  }
  IPrimaryMesh* toPrimaryMesh() override { return this; }

 public:

  IUserDataList* userDataList() override
  {
    _error();
    return nullptr;
  }
  const IUserDataList* userDataList() const override
  {
    _error();
    return nullptr;
  }

 public:

  IGhostLayerMng* ghostLayerMng() const override
  {
    _error();
    return nullptr;
  }
  IMeshUniqueIdMng* meshUniqueIdMng() const override
  {
    _error();
    return nullptr;
  }
  IMeshChecker* checker() const override
  {
    _error();
    return nullptr;
  }
  const MeshPartInfo& meshPartInfo() const override
  {
    _error();
    auto var = new MeshPartInfo{};
    return *var;
  }
  bool useMeshItemFamilyDependencies() const override
  {
    _error();
    return false;
  }
  IItemFamilyNetwork* itemFamilyNetwork() override
  {
    _error();
    return nullptr;
  }
  IIndexedIncrementalItemConnectivityMng* indexedConnectivityMng() override
  {
    _error();
    return nullptr;
  }

 public:

  IMeshCompactMng* _compactMng() override
  {
    _error();
    return nullptr;
  }
  InternalConnectivityPolicy _connectivityPolicy() const override
  {
    _error();
    return InternalConnectivityPolicy{};
  }

 public:

  IMeshMng* meshMng() const override
  {
    _error();
    return nullptr;
  }
  IVariableMng* variableMng() const override
  {
    _error();
    return nullptr;
  }
  ItemTypeMng* itemTypeMng() const override
  {
    _error();
    return nullptr;
  }

  IMeshInternal* _internalApi() override
  {
    ARCANE_THROW(NotImplementedException,"");
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_EMPTYMESH_H
