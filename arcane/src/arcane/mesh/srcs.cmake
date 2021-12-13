set( ARCANE_SOURCES
  AbstractItemFamilyTopologyModifier.cc
  AbstractItemFamilyTopologyModifier.h
  CompactIncrementalItemConnectivity.h
  CommonItemGroupFilterer.h
  CommonItemGroupFilterer.cc
  DynamicMesh.cc
  DynamicMesh.h
  DynamicMeshKindInfos.cc
  DynamicMeshKindInfos.h
  DynamicMeshIncrementalBuilder.cc
  DynamicMeshIncrementalBuilder.h
  DynamicMeshChecker.cc
  DynamicMeshChecker.h
  DynamicMeshMerger.cc
  DynamicMeshMerger.h
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
  ItemConnectivityInfo.cc
  ItemConnectivityInfo.h
  ItemConnectivitySelector.cc
  ItemConnectivitySelector.h
  ItemGroupsSynchronize.cc
  ItemGroupsSynchronize.h
  ItemGroupsSerializer2.cc
  ItemGroupsSerializer2.h
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
  FaceReorienter.cc
  FaceReorienter.h
  MeshCompacter.cc
  MeshCompacter.h
  MeshCompactMng.cc
  MeshCompactMng.h
  MeshExchange.cc
  MeshExchange.h
  MeshExchanger.cc
  MeshExchanger.h
  MeshExchangeMng.cc
  MeshExchangeMng.h
  MeshNodeMerger.cc
  MeshNodeMerger.h
  MeshVariables.cc
  MeshVariables.h
  MeshPartitionConstraintMng.cc
  MeshPartitionConstraintMng.h
  ExternalPartitionConstraint.cc
  ExternalPartitionConstraint.h
  ParticleFamily.cc
  ParticleFamily.h
  ParticleFamilyPolicyMng.cc
  BasicParticleExchanger.cc
  BasicParticleExchanger.h
  BasicParticleExchangerSerializer.cc
  BasicParticleExchangerSerializer.h
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
  ItemProperty.h
  IItemConnectivityGhostPolicy.h
  NewItemOwnerBuilder.h
  ConnectivityNewWithDependenciesTypes.h
  NewWithLegacyConnectivity.h
  MeshInfos.h
  PolyhedralMesh.cc
  PolyhedralMesh.h
  PolyhedralMeshService.cc
  GraphBuilder.h
  GraphDoFs.h
  GraphDoFs.cc
  )

set(AXL_FILES 
  BasicParticleExchanger
  PolyhedralMesh
  )
