// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* InterfaceImpl.cc                                            (C) 2000-2023 */
/*                                                                           */
/* pour les interfaces.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Sous Windows, il faut inclure les interfaces pour que leurs symboles
 * soient disponibles (sous linux aussi si on utilise gcc avec les infos
 * de visibilités).
 */

#include "arcane/utils/String.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/IArcaneMain.h"
#include "arcane/IServiceInfo.h"
#include "arcane/IBase.h"

#include "arcane/Assertion.h"
#include "arcane/IBase.h"
#include "arcane/IBackwardMng.h"
#include "arcane/ICaseMng.h"
#include "arcane/ICaseMeshReader.h"
#include "arcane/ICaseMeshService.h"
#include "arcane/ICaseMeshMasterService.h"
#include "arcane/IModuleMng.h"
#include "arcane/IServiceMng.h"
#include "arcane/ICodeService.h"
#include "arcane/ISubDomain.h"
#include "arcane/IServiceInfo.h"
#include "arcane/IService.h"
#include "arcane/IApplication.h"
#include "arcane/IMainFactory.h"
#include "arcane/IMeshBuilder.h"
#include "arcane/IMeshCompactMng.h"
#include "arcane/IMeshExchangeMng.h"
#include "arcane/IMeshCompacter.h"
#include "arcane/IMeshExchanger.h"
#include "arcane/IMeshFactory.h"
#include "arcane/IMeshFactoryMng.h"
#include "arcane/IMeshMng.h"
#include "arcane/IMeshPartitioner.h"
#include "arcane/IGridMeshPartitioner.h"
#include "arcane/IDirectExecution.h"
#include "arcane/IDirectSubDomainExecuteFunctor.h"
#include "arcane/ISerializer.h"
#include "arcane/IDeflateService.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/ItemTypes.h"
#include "arcane/IIndexedIncrementalItemConnectivity.h"
#include "arcane/IIndexedIncrementalItemConnectivityMng.h"
#include "arcane/IIncrementalItemConnectivity.h"
#include "arcane/IItemConnectivityAccessor.h"
#include "arcane/IItemConnectivityInfo.h"
#include "arcane/IItemConnectivity.h"
#include "arcane/IItemConnectivityMng.h"
#include "arcane/IItemConnectivitySynchronizer.h"
#include "arcane/ItemFamilyCompactInfos.h"
#include "arcane/IItemFamilyCompactPolicy.h"
#include "arcane/IItemFamilySerializer.h"
#include "arcane/IItemFamilySerializeStep.h"
#include "arcane/IItemFamilyExchanger.h"
#include "arcane/IItemFamilyModifier.h"
#include "arcane/IItemFamilyPolicyMng.h"
#include "arcane/IItemFamilyTopologyModifier.h"
#include "arcane/IDoFFamily.h"
#include "arcane/IParticleFamily.h"
#include "arcane/ItemFamilySerializeArgs.h"
#include "arcane/ITimeStats.h"
#include "arcane/ITimerMng.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IEntryPoint.h"
#include "arcane/ICaseOptions.h"
#include "arcane/Configuration.h"
#include "arcane/ConnectivityItemVector.h"
#include "arcane/IVariableFilter.h"
#include "arcane/IAsyncParticleExchanger.h"
#include "arcane/IParticleExchanger.h"
#include "arcane/IParallelExchanger.h"
#include "arcane/ITimeHistoryCurveWriter.h"
#include "arcane/IItemOperationByBasicType.h"
#include "arcane/IVariableSynchronizer.h"
#include "arcane/IVariableUtilities.h"
#include "arcane/IPhysicalUnitSystemService.h"
#include "arcane/IPhysicalUnitSystem.h"
#include "arcane/IPhysicalUnitConverter.h"
#include "arcane/IPhysicalUnit.h"
#include "arcane/IStandardFunction.h"
#include "arcane/IServiceAndModuleFactoryMng.h"
#include "arcane/IGhostLayerMng.h"
#include "arcane/IMeshUniqueIdMng.h"
#include "arcane/VariableStatusChangedEventArgs.h"
#include "arcane/MeshPartInfo.h"
#include "arcane/IGraph2.h"
#include "arcane/IGraphModifier2.h"
#include "arcane/IRandomNumberGenerator.h"
#include "arcane/ISimpleTableComparator.h"
#include "arcane/ISimpleTableInternalComparator.h"
#include "arcane/ISimpleTableInternalMng.h"
#include "arcane/ISimpleTableOutput.h"
#include "arcane/ISimpleTableReaderWriter.h"
#include "arcane/ISimpleTableWriterHelper.h"
#include "arcane/core/IMeshInitialAllocator.h"
#include "arcane/core/internal/IItemFamilyInternal.h"
#include "arcane/core/internal/IMeshInternal.h"

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

void IIncrementalItemConnectivity::
reserveMemoryForNbSourceItems([[maybe_unused]] Int32 n,
                              [[maybe_unused]] bool pre_alloc_connectivity)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
