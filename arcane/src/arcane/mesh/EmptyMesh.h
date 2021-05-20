// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EmptyMesh                                     (C) 2000-2021             */
/*                                                                           */
/* Brief code description                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_EMPTYMESH_H
#define ARCANE_EMPTYMESH_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMesh.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/VariableTypedef.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/IGraph.h"
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
#include "arcane/utils/IUserDataList.h"
#include "arcane/utils/IUserData.h"
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


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EmptyMesh : public IMesh
{

 public:
  ITraceMng* m_trace_mng;
  virtual ~EmptyMesh() = default;

 private:
  void _error() const { m_trace_mng->fatal() << "Using EmptyMesh"; }

 public:
  virtual const MeshHandle& handle() const {  _error(); auto var = new MeshHandle{}; return *var; }

 public:

  virtual String name() const { _error(); return String{}; }
  virtual Integer nbNode() { _error(); return -1; }
  virtual Integer nbEdge() { _error(); return -1; }
  virtual Integer nbFace() { _error(); return -1; }
  virtual Integer nbCell() { _error(); return -1; }
  virtual Integer nbItem(eItemKind ik) { _error(); return -1; }
  virtual ITraceMng* traceMng() { _error(); return nullptr; }
  virtual Integer dimension()   { _error(); return -1; }
  virtual NodeGroup allNodes() { _error(); return NodeGroup{}; }
  virtual EdgeGroup allEdges() { _error(); return EdgeGroup{}; }
  virtual FaceGroup allFaces() { _error(); return FaceGroup{}; }
  virtual CellGroup allCells() { _error(); return CellGroup{}; }
  virtual NodeGroup ownNodes() { _error(); return NodeGroup{}; }
  virtual EdgeGroup ownEdges() { _error(); return EdgeGroup{}; }
  virtual FaceGroup ownFaces() { _error(); return FaceGroup{}; }
  virtual CellGroup ownCells() { _error(); return CellGroup{}; }
  virtual FaceGroup outerFaces() { _error(); return FaceGroup{}; }

 public:

  virtual IItemFamily* createItemFamily(eItemKind ik,const String& name) { _error(); return nullptr; }
  virtual IItemFamily* findItemFamily(eItemKind ik,const String& name,bool create_if_needed=false) { _error(); return nullptr; }
  virtual IItemFamily* findItemFamily(const String& name,bool throw_exception=false) { _error(); return nullptr; }
  virtual IItemFamilyModifier* findItemFamilyModifier(eItemKind ik,const String& name) { _error(); return nullptr; }
  virtual IItemFamily* itemFamily(eItemKind ik) { _error(); return nullptr; }
  virtual IItemFamily* nodeFamily() { _error(); return nullptr; }
  virtual IItemFamily* edgeFamily() { _error(); return nullptr; }
  virtual IItemFamily* faceFamily() { _error(); return nullptr; }
  virtual IItemFamily* cellFamily() { _error(); return nullptr; }
  virtual IItemFamilyCollection itemFamilies() { _error(); return IItemFamilyCollection{}; }

 public:

  virtual void build() override { _error(); }
  virtual String factoryName() const override { _error(); return String{};}
  virtual ItemInternalList itemsInternal(eItemKind) override { _error(); return ItemInternalList{};}
  virtual SharedVariableNodeReal3 sharedNodesCoordinates() override { _error();  return SharedVariableNodeReal3{}; }
  virtual void checkValidMesh() override { _error(); }
  virtual void checkValidMeshFull() override { _error(); }
  virtual void synchronizeGroupsAndVariables() override { _error(); }

 public:

  virtual bool isAllocated() override { _error(); return false; }
  virtual Int64 timestamp() override { _error(); return -1; }

 public:

  virtual ISubDomain* subDomain() override { _error(); return nullptr; }

 public:

  virtual IParallelMng* parallelMng() override { _error(); return nullptr; }

 public:

  virtual IGraph* graph() override { _error(); return nullptr; }

 public:

  virtual VariableScalarInteger connectivity() override { _error(); return VariableScalarInteger{ nullptr}; }

  virtual CellGroup allActiveCells() override { _error(); return CellGroup{}; }
  virtual CellGroup ownActiveCells() override { _error(); return CellGroup{}; }
  virtual CellGroup allLevelCells(const Integer& level) override { _error(); return CellGroup{}; }
  virtual CellGroup ownLevelCells(const Integer& level) override { _error(); return CellGroup{}; }
  virtual FaceGroup allActiveFaces() override { _error(); return FaceGroup{}; }
  virtual FaceGroup ownActiveFaces() override { _error(); return FaceGroup{}; }
  virtual FaceGroup innerActiveFaces() override { _error(); return FaceGroup{}; }
  virtual FaceGroup outerActiveFaces() override { _error(); return FaceGroup{}; }

 public:

  virtual ItemGroupCollection groups() override { _error(); return ItemGroupCollection{}; }
  virtual ItemGroup findGroup(const String& name) override { _error(); return ItemGroup{}; }
  virtual void destroyGroups() override { _error();}

 public:

  virtual MeshItemInternalList* meshItemInternalList() override { _error(); return nullptr; }

 public:

  virtual void updateGhostLayers(bool remove_old_ghost) override { _error();}
  virtual void serializeCells(ISerializer* buffer,Int32ConstArrayView cells_local_id) override { _error();}
  virtual void prepareForDump() override { _error();}
  virtual void initializeVariables(const XmlNode& init_node) override { _error();}
  virtual void setCheckLevel(Integer level) override { _error();}
  virtual Integer checkLevel() const override { _error(); return -1; }
  virtual bool isDynamic() const override { _error(); return false; }
  virtual bool isAmrActivated() const override { _error(); return false; }

 public:

  virtual void computeTiedInterfaces(const XmlNode& mesh_node) override { _error();}
  virtual bool hasTiedInterface()  override { _error(); return false; }
  virtual TiedInterfaceCollection tiedInterfaces() override { _error(); return TiedInterfaceCollection{}; }
  virtual IMeshPartitionConstraintMng* partitionConstraintMng() override { _error(); return nullptr; }

 public:
  virtual IMeshUtilities* utilities() override { _error(); return nullptr; }
  virtual Properties* properties() override { _error(); return nullptr; }

 public:

  virtual IMeshModifier* modifier() override { _error(); return nullptr; }

 public:

  virtual VariableNodeReal3& nodesCoordinates() override { _error(); auto var = new VariableNodeReal3{ nullptr}; return *var;}
  virtual void defineParentForBuild(IMesh * mesh, ItemGroup group) override { _error(); }
  virtual IMesh * parentMesh() const override { _error(); return nullptr; }
  virtual ItemGroup parentGroup() const override { _error(); return ItemGroup{}; }
  virtual void addChildMesh(IMesh * sub_mesh) override { _error(); }
  virtual MeshCollection childMeshes() const override { _error(); return MeshCollection{}; }

 public:

  virtual bool isPrimaryMesh() const override { _error(); return false; }
  virtual IPrimaryMesh* toPrimaryMesh() override { _error(); return nullptr; }

 public:

  virtual IUserDataList* userDataList() override { _error(); return nullptr; }
  virtual const IUserDataList* userDataList() const override { _error(); return nullptr; }

 public:

  virtual IGhostLayerMng* ghostLayerMng() const override { _error(); return nullptr; }
  virtual IMeshChecker* checker() const override { _error(); return nullptr; }
  virtual const MeshPartInfo& meshPartInfo() const override { _error(); auto var=new MeshPartInfo{}; return *var; }
  virtual IItemFamilyNetwork* itemFamilyNetwork() override { _error(); return nullptr; }

 public:

  virtual IMeshCompactMng* _compactMng() override { _error(); return nullptr; }
  virtual InternalConnectivityPolicy _connectivityPolicy() const override { _error(); return InternalConnectivityPolicy{}; }

 public:

  virtual IMeshMng* meshMng() const override {  _error(); return nullptr; }
  virtual IVariableMng* variableMng() const override {  _error(); return nullptr; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_EMPTYMESH_H
