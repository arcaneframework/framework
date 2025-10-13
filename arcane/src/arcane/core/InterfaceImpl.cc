// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* InterfaceImpl.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Implémentation des interfaces.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Sous Windows, il faut inclure les interfaces pour que leurs symboles
 * soient disponibles (sous linux aussi si on utilise gcc avec les infos
 * de visibilités).
 */

#include "arcane/core/IVariableSynchronizerMng.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/IArcaneMain.h"
#include "arcane/core/IServiceInfo.h"
#include "arcane/core/Assertion.h"
#include "arcane/core/IBase.h"
#include "arcane/core/IBackwardMng.h"
#include "arcane/core/ICaseFunction.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/ICaseMeshReader.h"
#include "arcane/core/ICaseMeshService.h"
#include "arcane/core/ICaseMeshMasterService.h"
#include "arcane/core/ICheckpointMng.h"
#include "arcane/core/ICheckpointReader.h"
#include "arcane/core/ICheckpointWriter.h"
#include "arcane/core/IDataReader.h"
#include "arcane/core/IDataReader2.h"
#include "arcane/core/IDataReaderWriter.h"
#include "arcane/core/IDataWriter.h"
#include "arcane/core/IExtraGhostItemsBuilder.h"
#include "arcane/core/IModuleMng.h"
#include "arcane/core/IServiceMng.h"
#include "arcane/core/ICodeService.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IServiceInfo.h"
#include "arcane/core/IService.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IMeshBuilder.h"
#include "arcane/core/IMeshCompactMng.h"
#include "arcane/core/IMeshExchangeMng.h"
#include "arcane/core/IMeshCompacter.h"
#include "arcane/core/IMeshExchanger.h"
#include "arcane/core/IMeshFactory.h"
#include "arcane/core/IMeshFactoryMng.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IGridMeshPartitioner.h"
#include "arcane/core/IDataStorageFactory.h"
#include "arcane/core/IDirectExecution.h"
#include "arcane/core/IDirectSubDomainExecuteFunctor.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/IDeflateService.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/IInitialPartitioner.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IIndexedIncrementalItemConnectivity.h"
#include "arcane/core/IIndexedIncrementalItemConnectivityMng.h"
#include "arcane/core/IIncrementalItemConnectivity.h"
#include "arcane/core/IItemConnectivityAccessor.h"
#include "arcane/core/IItemConnectivityInfo.h"
#include "arcane/core/IItemConnectivity.h"
#include "arcane/core/IItemConnectivityMng.h"
#include "arcane/core/IItemConnectivitySynchronizer.h"
#include "arcane/core/ItemFamilyCompactInfos.h"
#include "arcane/core/ItemFamilyItemListChangedEventArgs.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IItemFamilyCompactPolicy.h"
#include "arcane/core/IItemFamilySerializer.h"
#include "arcane/core/IItemFamilySerializeStep.h"
#include "arcane/core/IItemFamilyExchanger.h"
#include "arcane/core/IItemFamilyModifier.h"
#include "arcane/core/IItemFamilyPolicyMng.h"
#include "arcane/core/IItemFamilyTopologyModifier.h"
#include "arcane/core/IDoFFamily.h"
#include "arcane/core/IParticleFamily.h"
#include "arcane/core/ItemFamilySerializeArgs.h"
#include "arcane/core/ITimeStats.h"
#include "arcane/core/ITimerMng.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IEntryPoint.h"
#include "arcane/core/ICaseOptions.h"
#include "arcane/core/ICaseFunctionProvider.h"
#include "arcane/core/IVariableSynchronizerMng.h"
#include "arcane/core/Configuration.h"
#include "arcane/core/ConnectivityItemVector.h"
#include "arcane/core/IVariableFilter.h"
#include "arcane/core/IVariableParallelOperation.h"
#include "arcane/core/IAsyncParticleExchanger.h"
#include "arcane/core/IParticleExchanger.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ITimeHistoryCurveWriter.h"
#include "arcane/core/ITimeHistoryCurveWriter2.h"
#include "arcane/core/ITimeHistoryTransformer.h"
#include "arcane/core/IItemOperationByBasicType.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/IVariableUtilities.h"
#include "arcane/core/IPhysicalUnitSystemService.h"
#include "arcane/core/IPhysicalUnitSystem.h"
#include "arcane/core/IPhysicalUnitConverter.h"
#include "arcane/core/IPhysicalUnit.h"
#include "arcane/core/IStandardFunction.h"
#include "arcane/core/ItemPairGroup.h"
#include "arcane/core/CaseFunction2.h"
#include "arcane/core/IServiceAndModuleFactoryMng.h"
#include "arcane/core/IGetVariablesValuesParallelOperation.h"
#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/IMeshUniqueIdMng.h"
#include "arcane/core/VariableStatusChangedEventArgs.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/IGraph2.h"
#include "arcane/core/IGraphModifier2.h"
#include "arcane/core/IRandomNumberGenerator.h"
#include "arcane/core/ISimpleTableComparator.h"
#include "arcane/core/ISimpleTableInternalComparator.h"
#include "arcane/core/ISimpleTableInternalMng.h"
#include "arcane/core/ISimpleTableOutput.h"
#include "arcane/core/ISimpleTableReaderWriter.h"
#include "arcane/core/ISimpleTableWriterHelper.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/MeshEvents.h"
#include "arcane/core/IExternalPlugin.h"
#include "arcane/core/IMeshSubdivider.h"

#include "arcane/core/IMeshInitialAllocator.h"
#include "arcane/core/internal/IItemFamilyInternal.h"
#include "arcane/core/internal/IMeshInternal.h"
#include "arcane/core/internal/IVariableInternal.h"
#include "arcane/core/internal/IMeshModifierInternal.h"
#include "arcane/core/internal/IVariableMngInternal.h"
#include "arcane/core/internal/IVariableSynchronizerMngInternal.h"
#include "arcane/core/internal/IIncrementalItemConnectivityInternal.h"
#include "arcane/core/internal/IPolyhedralMeshModifier.h"
#include "arcane/core/internal/IItemFamilySerializerMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IArcaneMain* IArcaneMain::global_arcane_main = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IArcaneMain* IArcaneMain::
arcaneMain()
{
  return global_arcane_main;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IArcaneMain::
setArcaneMain(IArcaneMain* arcane_main)
{
  global_arcane_main = arcane_main;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" String
arcaneNamespaceURI()
{
  return String("http://www.cea.fr/arcane/1.0");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IDeflateService::
compress(Span<const Byte> values,ByteArray& compressed_values)
{
  return compress(values.smallView(),compressed_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IDeflateService::
decompress(Span<const Byte> compressed_values,Span<Byte> values)
{
  return decompress(compressed_values.smallView(),values.smallView());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* IMeshPartitioner::
primaryMesh()
{
  IPrimaryMesh* primary_mesh = this->mesh()->toPrimaryMesh();
  ARCANE_CHECK_POINTER(primary_mesh);
  return primary_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* IItemFamilyModifier::
allocOne(Int64 uid,ItemTypeInfo* type, mesh::MeshInfos& mesh_info)
{
  return ItemCompatibility::_itemInternal(allocOne(uid,type->itemTypeId(),mesh_info));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* IItemFamilyModifier::
findOrAllocOne(Int64 uid,ItemTypeInfo* type, mesh::MeshInfos& mesh_info, bool& is_alloc)
{
  return ItemCompatibility::_itemInternal(findOrAllocOne(uid,type->itemTypeId(),mesh_info,is_alloc));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IIncrementalItemSourceConnectivity::
reserveMemoryForNbSourceItems([[maybe_unused]] Int32 n,
                              [[maybe_unused]] bool pre_alloc_connectivity)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IIncrementalItemSourceConnectivity::
_internalNotifySourceItemsAdded(Int32ConstArrayView local_ids)
{
  for (Int32 lid : local_ids)
    notifySourceItemAdded(ItemLocalId(lid));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IIncrementalItemConnectivity::
setConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids)
{
  removeConnectedItems(source_item);
  for (Int32 x : target_local_ids)
    addConnectedItem(source_item, ItemLocalId{ x });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IPostProcessorWriter::
setMesh([[maybe_unused]] IMesh* mesh)
{
  // Utiliser variable d'environnement.
  if (platform::getEnvironmentVariable("ARCANE_ALLOW_POSTPROCESSOR_SETMESH")=="1")
    return;
  ARCANE_FATAL("This call is deprecated and does not do anything."
               " You can temporarely disable this exception if you set the environment"
               " variable ARCANE_ALLOW_POSTPROCESSOR_SETMESH to '1'");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IMeshModifier::
addCells(const MeshModifierAddCellsArgs& args)
{
  addCells(args.nbCell(),args.cellInfos(),args.cellLocalIds());
}

void IMeshModifier::
addFaces(const MeshModifierAddFacesArgs& args)
{
  addFaces(args.nbFace(),args.faceInfos(),args.faceLocalIds());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IVariable::
synchronize(Int32ConstArrayView)
{
  ARCANE_THROW(NotImplementedException,"synchronize() with specific local ids");
}

void IItemFamily::
synchronize(VariableCollection, Int32ConstArrayView)
{
  ARCANE_THROW(NotImplementedException,"synchronize() with specific local ids");
}

void IVariableSynchronizer::
synchronize(IVariable*, Int32ConstArrayView)
{
  ARCANE_THROW(NotImplementedException,"synchronize() with specific local ids");
}

void IVariableSynchronizer::
synchronize(VariableCollection, Int32ConstArrayView)
{
  ARCANE_THROW(NotImplementedException,"synchronize() with specific local ids");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IMeshUtilities::
computeAdjacency(const ItemPairGroup& adjacency_array, eItemKind link_kind, Integer nb_layer)
{
  computeAdjency(adjacency_array, link_kind, nb_layer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup IItemFamily::
findAdjacencyItems(const ItemGroup& group,
                   const ItemGroup& sub_group,
                   eItemKind link_kind,
                   Integer nb_layer)
{
  return findAdjencyItems(group, sub_group, link_kind, nb_layer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IItemFamilyTopologyModifier::
setBackAndFrontCells(FaceLocalId, CellLocalId, CellLocalId)
{
  ARCANE_THROW(NotSupportedException, "only supported for FaceFamily of unstructured mesh");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
