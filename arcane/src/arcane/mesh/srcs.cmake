set( ARCANE_SOURCES
  # Les fichiers suivants sont en premier car ce sont
  # les plus longs à compiler
  PolyhedralMesh.cc
  PolyhedralMesh.h
  DynamicMesh.cc
  DynamicMesh.h

  AbstractItemFamilyTopologyModifier.cc
  AbstractItemFamilyTopologyModifier.h
  CartesianFaceUniqueIdBuilder.cc
  CompactIncrementalItemConnectivity.h
  CommonItemGroupFilterer.h
  CommonItemGroupFilterer.cc
  DynamicMeshKindInfos.cc
  DynamicMeshKindInfos.h
  DynamicMeshIncrementalBuilder.cc
  DynamicMeshIncrementalBuilder.h
  DynamicMeshCartesianBuilder.cc
  DynamicMeshChecker.cc
  DynamicMeshChecker.h
  DynamicMeshMerger.cc
  DynamicMeshMerger.h
  DualUniqueIdMng.h
  FaceUniqueIdBuilder.cc
  FaceUniqueIdBuilder.h
  FaceUniqueIdBuilder2.cc
  EdgeUniqueIdBuilder.cc
  EdgeUniqueIdBuilder.h
  GhostLayerBuilder.cc
  GhostLayerBuilder.h
  GhostLayerBuilder2.cc
  FullItemInfo.cc
  FullItemInfo.h
  OneMeshItemAdder.cc
  OneMeshItemAdder.h
  ExtraGhostCellsBuilder.cc
  ExtraGhostCellsBuilder.h
  ExtraGhostParticlesBuilder.cc
  ExtraGhostParticlesBuilder.h
  ExtraGhostItemsManager.cc
  ExtraGhostItemsManager.h
  IndexedIncrementalItemConnectivityMng.h
  IndexedIncrementalItemConnectivityMng.cc
  ItemConnectivityInfo.cc
  ItemConnectivityInfo.h
  ItemConnectivitySelector.cc
  ItemConnectivitySelector.h
  ItemGroupsSynchronize.cc
  ItemGroupsSynchronize.h
  ItemGroupsSerializer2.cc
  ItemGroupsSerializer2.h
  ItemsOwnerBuilder.h
  ItemsOwnerBuilder.cc
  ItemsExchangeInfo2.cc
  ItemsExchangeInfo2.h
  ItemGroupDynamicMeshObserver.cc
  ItemGroupDynamicMeshObserver.h
  ItemInternalConnectivityIndex.h
  BasicItemPairGroupComputeFunctor.cc
  BasicItemPairGroupComputeFunctor.h
  ItemFamily.cc
  ItemFamily.h
  ItemFamilyPolicyMng.cc
  ItemFamilyPolicyMng.h
  ItemFamilyCompactPolicy.cc
  ItemFamilyCompactPolicy.h
  ItemInternalMap.cc
  ItemInternalMap.h
  ItemSharedInfoList.cc
  ItemSharedInfoList.h
  ItemTools.cc
  ItemTools.h
  NodeFamily.cc
  NodeFamily.h
  EdgeFamily.cc
  EdgeFamily.h
  FaceFamily.cc
  FaceFamily.h
  CellFamily.cc
  CellFamily.h
  NodeFamilyPolicyMng.cc
  EdgeFamilyPolicyMng.cc
  FaceFamilyPolicyMng.cc
  CellFamilyPolicyMng.cc
  DoFFamilyPolicyMng.cc
  IndirectItemFamilySerializer.cc
  IndirectItemFamilySerializer.h
  CellFamilySerializer.cc
  CellFamilySerializer.h
  ParticleFamilySerializer.cc
  ParticleFamilySerializer.h
  ItemFamilyVariableSerializer.cc
  ItemFamilyVariableSerializer.h
  CellMerger.cc
  CellMerger.h
  GhostLayerMng.cc
  GhostLayerMng.h
  FaceReorienter.h
  MeshCompacter.cc
  MeshCompacter.h
  MeshCompactMng.cc
  MeshCompactMng.h
  MeshEventsImpl.cc
  MeshEventsImpl.h
  MeshExchange.cc
  MeshExchange.h
  MeshExchanger.cc
  MeshExchanger.h
  MeshExchangeMng.cc
  MeshExchangeMng.h
  MeshNodeMerger.cc
  MeshNodeMerger.h
  MeshUniqueIdMng.cc
  MeshUniqueIdMng.h
  MeshVariables.cc
  MeshVariables.h
  MeshPartitionConstraintMng.cc
  MeshPartitionConstraintMng.h
  ExternalPartitionConstraint.h
  ParticleFamily.cc
  ParticleFamily.h
  ParticleFamilyPolicyMng.cc
  BasicParticleExchanger.cc
  BasicParticleExchanger.h
  NonBlockingParticleExchanger.cc
  NonBlockingParticleExchanger.h
  AsyncParticleExchanger.cc
  AsyncParticleExchanger.h
  TiedInterface.cc
  TiedInterface.h
  TiedInterfaceExchanger.cc
  TiedInterfaceExchanger.h
  TiedInterfaceMng.cc
  TiedInterfaceMng.h
  UnstructuredMeshUtilities.cc
  UnstructuredMeshUtilities.h
  ItemRefinement.cc
  ItemRefinement.h
  MeshRefinement.cc
  MeshRefinement.h
  MapCoordToUid.cc
  MapCoordToUid.h
  ParallelAMRConsistency.cc
  ParallelAMRConsistency.h
  SubMeshTools.cc
  SubMeshTools.h
  ItemFamilyNetwork.cc
  ItemFamilyNetwork.h
  ItemFamilySerializer.cc
  ItemFamilySerializer.h
  ItemData.cc
  ItemData.h
  DoFFamily.cc
  DoFFamily.h
  GhostLayerFromConnectivityComputer.cc
  GhostLayerFromConnectivityComputer.h
  IndexedItemConnectivityAccessor.h
  IncrementalItemConnectivity.cc
  IncrementalItemConnectivity.h
  ItemConnectivity.cc
  ItemConnectivity.h
  ItemConnectivityMng.cc
  ItemConnectivityMng.h
  ItemConnectivitySynchronizer.cc
  ItemConnectivitySynchronizer.h
  AbstractItemFamilyTopologyModifier.h
  CompactIncrementalItemConnectivity.h
  DynamicMesh.h
  DynamicMeshKindInfos.h
  DynamicMeshIncrementalBuilder.h
  DynamicMeshChecker.h
  DynamicMeshMerger.h
  FaceUniqueIdBuilder.h
  EdgeUniqueIdBuilder.h
  GhostLayerBuilder.h
  FullItemInfo.h
  OneMeshItemAdder.h
  ExtraGhostCellsBuilder.h
  ExtraGhostParticlesBuilder.h
  ExtraGhostItemsManager.h
  ItemConnectivityInfo.h
  ItemConnectivitySelector.h
  ItemGroupsSynchronize.h
  ItemGroupsSerializer2.h
  ItemsExchangeInfo2.h
  ItemGroupDynamicMeshObserver.h
  BasicItemPairGroupComputeFunctor.h
  ItemFamily.h
  ItemFamilyPolicyMng.h
  ItemFamilyCompactPolicy.h
  ItemInternalMap.h
  ItemSharedInfoList.h
  ItemTools.h
  NodeFamily.h
  EdgeFamily.h
  FaceFamily.h
  CellFamily.h
  IndirectItemFamilySerializer.h
  CellFamilySerializer.h
  ParticleFamilySerializer.h
  ItemFamilyVariableSerializer.h
  CellMerger.h
  GhostLayerMng.h
  FaceReorienter.h
  MeshCompacter.h
  MeshCompactMng.h
  MeshExchange.h
  MeshExchanger.h
  MeshExchangeMng.h
  MeshNodeMerger.h
  MeshVariables.h
  MeshPartitionConstraintMng.h
  ExternalPartitionConstraint.h
  ParticleFamily.h
  BasicParticleExchanger.h
  NonBlockingParticleExchanger.h
  TiedInterface.h
  TiedInterfaceExchanger.h
  TiedInterfaceMng.h
  UnstructuredMeshUtilities.h
  ItemRefinement.h
  MeshRefinement.h
  MapCoordToUid.h
  ParallelAMRConsistency.h
  SubMeshTools.h
  ItemFamilyNetwork.h
  ItemFamilySerializer.h
  ItemData.h
  DoFFamily.h
  GhostLayerFromConnectivityComputer.h
  IncrementalItemConnectivity.h
  ItemConnectivity.h
  ItemConnectivityMng.h
  ItemConnectivitySynchronizer.h
  MeshGlobal.h
  DoFManager.h
  DoFManager.cc
  ItemProperty.h
  IItemConnectivityGhostPolicy.h
  NewItemOwnerBuilder.h
  ConnectivityNewWithDependenciesTypes.h
  NewWithLegacyConnectivity.h
  MeshInfos.h
  GraphBuilder.h
  GraphDoFs.h
  GraphDoFs.cc
  )

set(AXL_FILES 
  BasicParticleExchanger
  )
