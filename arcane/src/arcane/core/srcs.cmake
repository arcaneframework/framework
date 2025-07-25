﻿set(CURRENT_SRC_PATH ${Arcane_SOURCE_DIR}/src)

set(ARCANE_MATERIALS_SOURCES
  materials/CellToAllEnvCellConverter.h
  materials/ConstituentItem.h
  materials/ConstituentItemBase.h
  materials/ConstituentItemSharedInfo.h
  materials/ConstituentItemLocalId.h
  materials/ConstituentItemLocalId.cc
  materials/MaterialsCoreGlobal.h
  materials/MaterialsCoreGlobal.cc
  materials/MatItem.h
  materials/MatVarIndex.h
  materials/MatVarIndex.cc
  materials/IEnumeratorTracer.h
  materials/IMeshMaterialVariable.cc
  materials/IMeshMaterialVariable.h
  materials/IScalarMeshMaterialVariable.h
  materials/IArrayMeshMaterialVariable.h
  materials/IMeshMaterialMng.cc
  materials/IMeshMaterialMng.h
  materials/IMeshBlock.h
  materials/IMeshComponent.h
  materials/IMeshEnvironment.h
  materials/IMeshMaterial.h
  materials/IMeshMaterialVariableFactoryMng.h
  materials/IMeshMaterialVariableFactory.h
  materials/ComponentItem.cc
  materials/ComponentItem.h
  materials/ComponentItemInternal.cc
  materials/ComponentItemInternal.h
  materials/ComponentItemVector.cc
  materials/ComponentItemVector.h
  materials/ComponentItemVectorView.cc
  materials/ComponentItemVectorView.h
  materials/ComponentPartItemVectorView.cc
  materials/ComponentPartItemVectorView.h
  materials/ConstituentItemLocalIdList.cc
  materials/EnvItemVector.cc
  materials/EnvItemVector.h
  materials/MatItemVector.cc
  materials/MatItemVector.h
  materials/MatItemEnumerator.cc
  materials/MatItemEnumerator.h
  materials/MaterialVariableBuildInfo.cc
  materials/MaterialVariableBuildInfo.h
  materials/MaterialVariableTypeInfo.cc
  materials/MaterialVariableTypeInfo.h
  materials/MeshMaterialVariableRef.cc
  materials/MeshMaterialVariableRef.h
  materials/MeshEnvironmentVariableRef.cc
  materials/MeshEnvironmentVariableRef.h
  materials/MeshMaterialVariableComputeFunction.h
  materials/IMeshMaterialVariableComputeFunction.h
  materials/internal/IMeshMaterialVariableInternal.h
  materials/internal/IMeshComponentInternal.h
  materials/internal/IMeshMaterialMngInternal.h
  materials/internal/ConstituentItemLocalIdList.h
  )

set(ARCANE_INTERNAL_SOURCES
  internal/ICaseMngInternal.h
  internal/IDataInternal.h
  internal/MshMeshGenerationInfo.h
  internal/CartesianMeshGenerationInfo.h
  internal/CartesianMeshGenerationInfo.cc
  internal/CartesianMeshAllocateBuildInfoInternal.h
  internal/UnstructuredMeshAllocateBuildInfoInternal.h
  internal/IParallelMngInternal.h
  internal/IItemFamilyInternal.h
  internal/IMeshInternal.h
  internal/IParallelMngUtilsFactory.h
  internal/IVariableInternal.h
  internal/IMeshModifierInternal.h
  internal/ItemGroupImplInternal.h
  internal/ItemGroupInternal.h
  internal/ICaseOptionListInternal.h
  internal/IVariableMngInternal.h
  internal/IVariableSynchronizerMngInternal.h
  internal/StringVariableReplace.h
  internal/StringVariableReplace.cc
  internal/ITimeHistoryMngInternal.h
  internal/VariableUtilsInternal.h
  internal/IPolyhedralMeshModifier.h
  internal/SerializeMessage.h
  internal/VtkCellTypes.h
  internal/ParallelMngInternal.h
  internal/ParallelMngInternal.cc
  )

set(ARCANE_ORIGINAL_SOURCES
  # Les fichiers suivants sont en premier car ce sont
  # les plus longs à compiler
  Array2Variable.cc
  Array2Variable.h
  VariableArray.cc
  VariableArray.h
  MeshVariableTpl.cc
  MeshVariableTplArray.cc

  AbstractModule.h
  ApplicationBuildInfo.h
  ApplicationBuildInfo.cc
  ArcaneException.h
  ArcaneTypes.h
  ArcaneTypes.cc

  BasicModule.h

  CommonVariables.h

  DotNetRuntimeInitialisationInfo.h
  DotNetRuntimeInitialisationInfo.cc
  FaceReorienter.h
  FaceReorienter.cc

  IApplication.h
  IArcaneMain.h
  IBase.h
  ICaseFunction.h
  ICaseFunctionProvider.h
  ICaseMng.h
  ICaseOptions.h
  ICaseOptionList.h
  ICheckpointReader.h
  ICheckpointWriter.h
  ICriteriaLoadBalanceMng.h
  IData.h
  IDataReader.h
  IDataVisitor.h
  IDataWriter.h
  IDirectExecution.h
  IDirectory.h
  IEntryPoint.h
  IExternalPlugin.h
  IItemConnectivityInfo.h
  IItemFamily.h
  ISession.h
  IMainFactory.h
  IMesh.h
  IMeshBase.h
  IMeshInitialAllocator.h
  IMeshModifier.h
  IMeshReader.h
  IMeshSubdivider.h
  IMeshUtilities.h
  IModule.h
  IModuleFactory.h
  IPostProcessorWriter.h
  IPrimaryMesh.h
  ISerializedData.h
  IService.h
  IServiceFactory.h
  IServiceInfo.h
  ISubDomain.h
  ItemGroup.h
  ItemGroupImpl.h
  ItemPairGroup.h
  ItemPairGroupBuilder.h
  ITimeHistoryCurveWriter.h
  ITimeHistoryCurveWriter2.h
  ITimeHistoryMng.h
  ITimeHistoryAdder.h
  ITimeLoopMng.h
  IUnitTest.h
  IVariable.h
  IVariableMng.h
  IVariableReader.h

  IAsyncParticleExchanger.h
  IBackwardMng.h
  ICartesianMeshGenerationInfo.h
  ICaseDocument.h
  ICaseDocumentVisitor.h
  ICaseMeshMasterService.h
  ICaseMeshReader.h
  ICaseMeshService.h
  ICheckpointMng.h
  ICodeService.h
  IConfiguration.h
  IConfigurationMng.h
  IConfigurationSection.h
  IDataFactory.h
  IDataFactoryMng.h
  IDataReader2.h
  IDataReaderWriter.h
  IDataStorageFactory.h
  IDeflateService.h
  IDirectSubDomainExecuteFunctor.h
  IDoFFamily.h
  IEntryPointMng.h
  IExtraGhostCellsBuilder.h
  IExtraGhostItemsBuilder.h
  IExtraGhostParticlesBuilder.h
  IFactoryService.h
  IGetVariablesValuesParallelOperation.h
  IGhostLayerMng.h
  IGraph2.h
  IGraphModifier2.h
  IGridMeshPartitioner.h
  IIOMng.h
  IIncrementalItemConnectivity.h
  IIndexedIncrementalItemConnectivity.h
  IIndexedIncrementalItemConnectivityMng.h
  IInitialPartitioner.h
  IItemConnectivity.h
  IItemConnectivityAccessor.h
  IItemConnectivityMng.h
  IItemConnectivitySynchronizer.h
  IItemEnumeratorTracer.h
  IItemFamilyCompactPolicy.h
  IItemFamilyExchanger.h
  IItemFamilyModifier.h
  IItemFamilyNetwork.h
  IItemFamilyPolicyMng.h
  IItemFamilySerializeStep.h
  IItemFamilySerializer.h
  IItemFamilyTopologyModifier.h
  IItemInternalSortFunction.h
  IItemOperationByBasicType.h
  ILoadBalanceMng.h
  IMeshArea.h
  IMeshBuilder.h
  IMeshChecker.h
  IMeshCompactMng.h
  IMeshCompacter.h
  IMeshExchangeMng.h
  IMeshExchanger.h
  IMeshFactory.h
  IMeshFactoryMng.h
  IMeshMng.h
  IMeshPartitionConstraint.h
  IMeshPartitionConstraintMng.h
  IMeshPartitioner.h
  IMeshPartitionerBase.h
  IMeshStats.h
  IMeshSubMeshTransition.h
  IMeshUniqueIdMng.h
  IMeshWriter.h
  IModuleMaster.h
  IModuleMng.h
  IObservable.h
  IObserver.h
  IParallelDispatch.h
  IParallelExchanger.h
  IParallelMng.h
  IParallelNonBlockingCollective.h
  IParallelNonBlockingCollectiveDispatch.h
  IParallelReplication.h
  IParallelSort.h
  IParallelSuperMng.h
  IParallelTopology.h
  IParticleExchanger.h
  IParticleFamily.h
  IPhysicalUnit.h
  IPhysicalUnitConverter.h
  IPhysicalUnitSystem.h
  IPhysicalUnitSystemService.h
  IProperty.h
  IPropertyMng.h
  IRandomNumberGenerator.h
  IRayMeshIntersection.h
  IRessourceMng.h
  ISerializeMessage.h
  ISerializeMessageList.h
  ISerializer.h
  IServiceAndModuleFactoryMng.h
  IServiceLoader.h
  IServiceMng.h
  ISharedReference.h
  ISimpleTableComparator.h
  ISimpleTableInternalComparator.h
  ISimpleTableInternalMng.h
  ISimpleTableOutput.h
  ISimpleTableReaderWriter.h
  ISimpleTableWriterHelper.h
  IStandardFunction.h
  ITiedInterface.h
  ITimeHistoryTransformer.h
  ITimeLoop.h
  ITimeLoopService.h
  ITimeStats.h
  ITimerMng.h
  ITransferValuesParallelOperation.h
  IVariableAccessor.h
  IVariableComputeFunction.h
  IVariableFactory.h
  IVariableFilter.h
  IVariableParallelOperation.h
  IVariableSynchronizer.h
  IVariableSynchronizerMng.h
  IVariableUtilities.h
  IVariableWriter.h
  IVerifierService.h
  IXmlDocumentHolder.h
  IndexedItemConnectivityView.cc
  IndexedItemConnectivityView.h
  InterfaceImpl.cc
  Item.cc
  Item.h
  ItemArrayEnumerator.h
  ItemCompare.h
  ItemCompatibility.h
  ItemConnectivityContainerView.cc
  ItemConnectivityContainerView.h
  ItemEnumerator.cc
  ItemEnumerator.h
  ItemEnumeratorBase.h
  ItemFamilyCompactInfos.h
  ItemFamilyItemListChangedEventArgs.h
  ItemFamilySerializeArgs.h
  ItemFlags.h
  ItemFunctor.cc
  ItemFunctor.h
  ItemGenericInfoListView.cc
  ItemGenericInfoListView.h
  ItemGroup.cc
  ItemGroupComputeFunctor.cc
  ItemGroupComputeFunctor.h
  ItemGroupImpl.cc
  ItemGroupInternal.cc
  ItemGroupObserver.h
  ItemGroupRangeIterator.cc
  ItemGroupRangeIterator.h
  ItemIndexArrayView.h
  ItemIndexedListView.h
  ItemInfoListView.cc
  ItemInfoListView.h
  ItemInternal.cc
  ItemInternal.h
  ItemInternalEnumerator.h
  ItemInternalSortFunction.h
  ItemInternalVectorView.h
  ItemLocalId.h
  ItemLocalId.cc
  ItemLocalIdListContainerView.h
  ItemLocalIdListView.h
  ItemLocalIdListView.cc
  ItemLoop.h
  ItemPairEnumerator.cc
  ItemPairEnumerator.h
  ItemPairGroup.cc
  ItemPairGroupBuilder.cc
  ItemPairGroupImpl.cc
  ItemPairGroupImpl.h
  ItemPrinter.cc
  ItemPrinter.h
  ItemRefinementPattern.cc
  ItemRefinementPattern.h
  ItemSharedInfo.cc
  ItemSharedInfo.h
  ItemTypeId.h
  ItemTypeInfo.cc
  ItemTypeInfo.h
  ItemTypeInfoBuilder.cc
  ItemTypeInfoBuilder.h
  ItemTypeMng.cc
  ItemTypeMng.h
  ItemTypes.h
  ItemUniqueId.h
  ItemVector.cc
  ItemVector.h
  ItemVectorView.cc
  ItemVectorView.h
  ItemConnectedListView.h
  ItemConnectedEnumeratorBase.h
  ItemConnectedEnumerator.h
  ItemAllocationInfo.h

  MeshHandle.h
  MeshPartInfo.h
  MeshReaderMng.h
  ModuleBuildInfo.h
  ModuleFactory.h

  MshMeshGenerationInfo.cc

  PrivateVariableScalar.h
  PrivateVariableArray.h

  ServiceBuilder.h
  ServiceBuildInfo.h
  ServiceFactory.h
  ServiceInfo.h
  ServiceInstance.h
  ServiceProperty.h
  SharedReference.h
  SharedVariable.h
  StandardCaseFunction.h
  StdNum.h

  VariableBuildInfo.h
  VariableCollection.h
  VariableRef.h
  VariableUtils.h
  VariableUtils.cc
  VariableUtilsInternal.cc

  XmlNode.h
  XmlNodeList.h

  AbstractCaseDocumentVisitor.cc
  AbstractCaseDocumentVisitor.h
  AbstractDataVisitor.cc
  AbstractDataVisitor.h
  AbstractItemOperationByBasicType.cc
  AbstractItemOperationByBasicType.h
  AbstractModule.cc
  AbstractService.cc
  AbstractService.h
  AcceleratorRuntimeInitialisationInfo.h
  Algorithm.h
  ArcaneException.cc
  ArcaneVersion.h
  Assertion.cc
  Assertion.h
  BasicModule.cc
  BasicService.cc
  BasicService.h
  BasicTimeLoopService.h
  BasicUnitTest.cc
  BasicUnitTest.h
  BlockIndexList.h
  BlockIndexList.cc
  CartesianGridDimension.h
  CartesianGridDimension.cc
  CartesianMeshAllocateBuildInfo.h
  CartesianMeshAllocateBuildInfo.cc
  CaseDatasetSource.cc
  CaseDatasetSource.h
  CaseFunction.cc
  CaseFunction.h
  CaseFunction2.h
  CaseOptions.h
  CaseOptionServiceImpl.h
  CaseOptionTypes.h
  CaseNodeNames.cc
  CaseNodeNames.h
  CaseOptionBase.h
  CaseOptionBase.cc
  CaseOptionBuildInfo.cc
  CaseOptionBuildInfo.h
  CaseOptionEnum.cc
  CaseOptionEnum.h
  CaseOptionComplexValue.cc
  CaseOptionComplexValue.h
  CaseOptionError.cc
  CaseOptionError.h
  CaseOptionException.cc
  CaseOptionException.h
  CaseOptionExtended.cc
  CaseOptionExtended.h
  CaseOptionList.cc
  CaseOptionService.cc
  CaseOptionService.h
  CaseOptionSimple.cc
  CaseOptionSimple.h
  CaseOptions.cc
  CaseOptionsMain.cc
  CaseOptionsMain.h
  CaseOptionsMulti.h
  CaseTable.cc
  CaseTable.h
  CaseTableParams.cc
  CaseTableParams.h
  CheckpointInfo.cc
  CheckpointInfo.h
  CheckpointService.cc
  CheckpointService.h
  AxlOptionsBuilder.h
  AxlOptionsBuilder.cc
  CodeService.cc
  CodeService.h
  CommonVariables.cc
  Concurrency.cc
  Concurrency.h
  Configuration.h
  ConfigurationPropertyReader.h
  Connectivity.cc
  Connectivity.h
  ConnectivityItemVector.h
  Data.cc
  DataTypeDispatchingDataVisitor.cc
  DataTypeDispatchingDataVisitor.h
  DataView.h
  Directory.cc
  Directory.h
  Dom.h
  DomDeclaration.h
  DomLibXml2V2.cc
  DomUtils.cc
  DomUtils.h
  EntryPoint.cc
  EntryPoint.h
  EnumeratorTraceWrapper.h
  ExternalPartitionConstraint.h
  ExternalPartitionConstraint.cc
  Factory.h
  FactoryService.cc
  FactoryService.h
  GeometricUtilities.cc
  GeometricUtilities.h
  GlobalTimeHistoryAdder.cc
  GlobalTimeHistoryAdder.h
  GroupIndexTable.cc
  GroupIndexTable.h
  MathUtils.cc
  MathUtils.h
  MeshAccessor.cc
  MeshAccessor.h
  MeshArea.cc
  MeshArea.h
  MeshAreaAccessor.cc
  MeshAreaAccessor.h
  MeshBuildInfo.cc
  MeshBuildInfo.h
  MeshCriteriaLoadBalanceMng.cc
  MeshCriteriaLoadBalanceMng.h
  MeshMDVariableRef.h
  MeshEvents.h
  MeshHandle.cc
  MeshKind.h
  MeshKind.cc
  MeshItemInternalList.cc
  MeshItemInternalList.h
  MeshPartInfo.cc
  MeshPartialVariableArrayRef.h
  MeshPartialVariableArrayRefT.H
  MeshPartialVariableScalarRef.h
  MeshPartialVariableScalarRefT.H
  MeshReaderMng.cc
  MeshStats.cc
  MeshStats.h
  MeshTimeHistoryAdder.cc
  MeshTimeHistoryAdder.h
  MeshToMeshTransposer.cc
  MeshToMeshTransposer.h
  MeshUtils.cc
  MeshUtils2.cc
  MeshUtils.h
  MeshVariable.h
  MeshVariableArrayRef.h
  MeshVariableArrayRefT.H
  MeshVariableInfo.h
  MeshVariableRef.cc
  MeshVariableRef.h
  MeshVariableScalarRef.h
  MeshVariableScalarRefT.H
  MeshVisitor.cc
  MeshVisitor.h
  ModuleBuildInfo.cc
  ModuleFactory.cc
  ModuleMaster.cc
  ModuleMaster.h
  ModuleProperty.h
  MultiArray2Variable.h
  MultiArray2VariableRef.h
  NodesOfItemReorderer.h
  NodesOfItemReorderer.cc
  MachineMemoryWindowBase.cc
  MachineMemoryWindowBase.h
  MachineMemoryWindow.h
  NullXmlDocumentHolder.cc
  Observable.h
  ObservablePool.h
  Observer.h
  ObserverPool.cc
  ObserverPool.h
  OutputChecker.cc
  OutputChecker.h
  Parallel.cc
  Parallel.h
  ParallelExchangerOptions.h
  ParallelMngDispatcher.cc
  ParallelMngDispatcher.h
  ParallelMngUtils.cc
  ParallelMngUtils.h
  ParallelNonBlockingCollectiveDispatcher.cc
  ParallelNonBlockingCollectiveDispatcher.h
  ParallelSuperMngDispatcher.cc
  ParallelSuperMngDispatcher.h
  PostProcessorWriterBase.cc
  PostProcessorWriterBase.h
  PreciseOutputChecker.cc
  PreciseOutputChecker.h
  PrivateVariableArrayT.H
  PrivateVariableArrayTpl.cc
  PrivateVariableScalarT.H
  PrivateVariableScalarTpl.cc
  Properties.cc
  Properties.h
  Property.cc
  RessourceMng.cc
  SequentialSection.cc
  SequentialSection.h
  SerializeBuffer.cc
  SerializeBuffer.h
  SerializeMessage.cc
  SerializedData.cc
  Service.h
  ServiceBuildInfo.cc
  ServiceBuilder.cc
  ServiceFactory.cc
  ServiceFinder.h
  ServiceFinder2.h
  ServiceInfo.cc
  ServiceOptions.h
  ServiceRegisterer.cc
  ServiceRegisterer.h
  ServiceUtils.h
  SharedReference.cc
  SimdItem.cc
  SimdItem.h
  SimdMathUtils.h
  SimpleProperty.h
  SimpleSVGMeshExporter.cc
  SimpleSVGMeshExporter.h
  SimpleTableInternal.h
  StandardCaseFunction.cc
  StringDictionary.h
  SubDomainBuildInfo.cc
  SubDomainBuildInfo.h
  SynchronizerMatrixPrinter.cc
  SynchronizerMatrixPrinter.h
  TemporaryVariableBuildInfo.cc
  TemporaryVariableBuildInfo.h
  TiedFace.h
  TiedNode.h
  TimeLoop.cc
  TimeLoop.h
  TimeLoopEntryPointInfo.h
  Timer.cc
  Timer.h
  UnitTestServiceAdapter.h
  UnstructuredMeshConnectivity.cc
  UnstructuredMeshConnectivity.h
  UnstructuredMeshAllocateBuildInfo.h
  UnstructuredMeshAllocateBuildInfo.cc
  Variable.cc
  Variable.h
  VariableAccessor.h
  VariableBuildInfo.cc
  VariableCollection.cc
  VariableComparer.h
  VariableComparer.cc
  VariableComputeFunction.h
  VariableDataTypeTraits.h
  VariableDependInfo.cc
  VariableDependInfo.h
  VariableDiff.h
  VariableDiff.cc
  VtkCellTypes.cc
  VariableFactory.cc
  VariableFactory.h
  VariableFactoryRegisterer.cc
  VariableFactoryRegisterer.h
  VariableInfo.cc
  VariableInfo.h
  VariableList.h
  VariableMetaData.cc
  VariableMetaData.h
  VariableRef.cc
  VariableRefArray.cc
  VariableRefArray.h
  VariableRefArray2.cc
  VariableRefArray2.h
  VariableRefArrayLock.h
  VariableRefScalar.cc
  VariableRefScalar.h
  VariableScalar.cc
  VariableScalar.h
  VariableStatusChangedEventArgs.h
  VariableSynchronizerEventArgs.cc
  VariableSynchronizerEventArgs.h
  VariableTypeInfo.cc
  VariableTypeInfo.h
  VariableTypedef.h
  VariableTypes.h
  VariableView.h
  VerifierService.cc
  VerifierService.h
  XmlException.h
  XmlNode.cc
  XmlNodeIterator.h
  XmlNodeList.cc
  XmlProperty.cc
  XmlProperty.h

  packages/Mesh.h
  packages/Variable.h

  datatype/ArrayVariant.cc
  datatype/ArrayVariant.h
  datatype/BadVariantTypeException.cc
  datatype/BadVariantTypeException.h
  datatype/DataAllocationInfo.h
  datatype/DataStorageBuildInfo.h
  datatype/DataStorageBuildInfo.cc
  datatype/DataStorageTypeInfo.h
  datatype/DataStorageTypeInfo.cc
  datatype/DataTracer.cc
  datatype/DataTracer.h
  datatype/DataTypes.cc
  datatype/DataTypes.h
  datatype/DataTypeTraits.h
  datatype/IDataOperation.h
  datatype/IDataTracer.h
  datatype/RealArrayVariant.cc
  datatype/RealArrayVariant.h
  datatype/RealArray2Variant.cc
  datatype/RealArray2Variant.h
  datatype/ScalarVariant.cc
  datatype/ScalarVariant.h
  datatype/SmallVariant.cc
  datatype/SmallVariant.h
  datatype/VariantBase.cc
  datatype/VariantBase.h

  # expr/IExpressionImpl.h
  # expr/Expression.h
  # expr/ArrayExpressionImpl.cc
  # expr/ArrayExpressionImpl.h
  # expr/BadExpressionException.cc
  # expr/BadExpressionException.h
  # expr/BadOperandException.cc
  # expr/BadOperandException.h
  # expr/BadOperationException.cc
  # expr/BadOperationException.h
  # expr/Expression.cc
  # expr/ExpressionResult.cc
  # expr/ExpressionResult.h
  # expr/ExpressionImpl.cc
  # expr/ExpressionImpl.h
  # expr/UnaryExpressionImpl.cc
  # expr/UnaryExpressionImpl.h
  # expr/LitteralExpressionImpl.cc
  # expr/LitteralExpressionImpl.h
  # expr/BinaryExpressionImpl.cc
  # expr/BinaryExpressionImpl.h
  # expr/WhereExpressionImpl.cc
  # expr/WhereExpressionImpl.h
  # expr/OperatorMng.cc
  # expr/OperatorMng.h

  anyitem/AnyItem.h
  anyitem/AnyItemGlobal.h
  anyitem/AnyItemPrivate.h
  anyitem/AnyItemArray.h
  anyitem/AnyItemArray2.h
  anyitem/AnyItemFamily.h
  anyitem/AnyItemFamilyObserver.h
  anyitem/AnyItemGroup.h
  anyitem/AnyItem.h
  anyitem/AnyItemLinkFamily.h
  anyitem/AnyItemLinkVariable.h
  anyitem/AnyItemLinkVariableArray.h
  anyitem/AnyItemUserGroup.h
  anyitem/AnyItemVariable.h
  anyitem/AnyItemVariableArray.h

  matvec/AMG.cc
  matvec/Matrix.cc
  matvec/Matrix.h
  matvec/Vector.cc
  matvec/Vector.h

  parallel/BitonicSort.h
  parallel/BitonicSortT.H
  parallel/GhostItemsVariableParallelOperation.cc
  parallel/GhostItemsVariableParallelOperation.h
  parallel/IMultiReduce.h
  parallel/IRequestList.h
  parallel/IStat.h
  parallel/MultiReduce.cc
  parallel/Stat.cc
  parallel/VariableParallelOperationBase.cc
  parallel/VariableParallelOperationBase.h

  random/ConstMod.h
  random/InversiveCongruential.h
  random/LinearCongruential.h
  random/MersenneTwister.h
  random/NormalDistribution.h
  random/RandomGlobal.h
  random/TKiss.h
  random/TMrg32k3a.h
  random/Uniform01.h
  random/UniformInt.h
  random/UniformOnSphere.h
  random/UniformSmallInt.h

  TimeLoopSingletonServiceInfo.h
  VarRefEnumerator.h
  RawCopy.h
  )

if (ARCANE_HAS_ACCELERATOR_API)
  list(APPEND ARCANE_ORIGINAL_SOURCES
    MeshMDVariableRef.cc
  )
endif()

set(ARCANE_SOURCES
  ${ARCANE_ORIGINAL_SOURCES}
  ${ARCANE_MATERIALS_SOURCES}
  ${ARCANE_INTERNAL_SOURCES}
)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
