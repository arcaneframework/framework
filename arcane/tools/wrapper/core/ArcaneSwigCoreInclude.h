// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneSwigCoreInclude.h                                     (C) 2000-2024 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_DOTNET
#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/List.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ISession.h"
#include "arcane/core/IDirectory.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IArcaneMain.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemPairGroupBuilder.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemInfoListView.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IParallelSuperMng.h"
#include "arcane/core/BasicModule.h"
#include "arcane/core/EntryPoint.h"
#include "arcane/core/ModuleFactory.h"
#include "arcane/core/ServiceRegisterer.h"
#include "arcane/core/IServiceInfo.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/ServiceBuildInfo.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IItemConnectivityInfo.h"
#include "arcane/core/ItemPairGroup.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/IExternalPlugin.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/CaseOptionBuildInfo.h"
#include "arcane/core/CaseOptionBase.h"
#include "arcane/core/CaseOptions.h"
#include "arcane/core/CaseOptionsMulti.h"
#include "arcane/core/CaseOptionService.h"
#include "arcane/core/StandardCaseFunction.h"
#include "arcane/core/CaseFunction2.h"
#include "arcane/core/ICaseOptions.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/ICaseFunctionProvider.h"
#include "arcane/core/ApplicationBuildInfo.h"

#include "arcane/core/MeshVariableRef.h"
#include "arcane/core/PrivateVariableScalar.h"
#include "arcane/core/MeshVariableScalarRef.h"
#include "arcane/core/MeshVariableArrayRef.h"
#include "arcane/core/MeshPartialVariableScalarRef.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/IMeshUtilities.h"

#include "arcane/core/ServiceInfo.h"

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/core/StdNum.h"
#include "arcane/datatype/ScalarVariant.h"
#include "arcane/datatype/ArrayVariant.h"
#include "arcane/datatype/DataStorageTypeInfo.h"
#include "arcane/datatype/DataAllocationInfo.h"

#include "arcane/impl/ArcaneMain.h"
#include "arcane/impl/ArcaneSimpleExecutor.h"

#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/XmlNodeIterator.h"

#include "arcane/core/Observer.h"
#include "arcane/core/IObservable.h"

#include "arcane/core/ITimeHistoryMng.h"

#include "arcane/core/IUnitTest.h"
#include "arcane/core/BasicUnitTest.h"

#include "arcane/core/ISerializedData.h"
#include "arcane/core/IDataReader.h"
#include "arcane/core/IDataWriter.h"
#include "arcane/core/IVariableReader.h"
#include "arcane/core/IDirectExecution.h"
#include "arcane/core/ICheckpointReader.h"
#include "arcane/core/ICheckpointWriter.h"
#include "arcane/core/ITimeHistoryCurveWriter.h"
#include "arcane/core/ITimeHistoryCurveWriter2.h"
#include "arcane/core/IMeshReader.h"

#include "arcane/core/SharedVariable.h"
#include "arcane/core/ArcaneException.h"

#include "arcane/core/AbstractDataVisitor.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/MeshReaderMng.h"
#include "arcane/core/DotNetRuntimeInitialisationInfo.h"
#include "arcane/utils/ExternalRef.h"
#include "arcane/accelerator/core/AcceleratorRuntimeInitialisationInfo.h"
#include "arcane/accelerator/core/DeviceId.h"

#include "ArcaneSwigUtils.h"

using namespace Arcane;
