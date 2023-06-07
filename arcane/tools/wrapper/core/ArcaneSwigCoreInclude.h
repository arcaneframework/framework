// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneSwigCoreInclude.h                                     (C) 2000-2022 */
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

#include "arcane/expr/Expression.h"
#include "arcane/expr/ArrayExpressionImpl.h"

#include "arcane/IApplication.h"
#include "arcane/ISubDomain.h"
#include "arcane/ISession.h"
#include "arcane/IDirectory.h"
#include "arcane/XmlNode.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IArcaneMain.h"
#include "arcane/IVariableMng.h"
#include "arcane/IVariable.h"
#include "arcane/ItemGroup.h"
#include "arcane/ItemPairGroupBuilder.h"
#include "arcane/VariableCollection.h"
#include "arcane/Item.h"
#include "arcane/ItemInfoListView.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/IParallelSuperMng.h"
#include "arcane/BasicModule.h"
#include "arcane/EntryPoint.h"
#include "arcane/ModuleFactory.h"
#include "arcane/ServiceRegisterer.h"
#include "arcane/IServiceInfo.h"
#include "arcane/FactoryService.h"
#include "arcane/ServiceBuildInfo.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/IItemFamily.h"
#include "arcane/IItemConnectivityInfo.h"
#include "arcane/ItemPairGroup.h"
#include "arcane/IParallelMng.h"
#include "arcane/IPostProcessorWriter.h"
#include "arcane/Parallel.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/CaseOptionBuildInfo.h"
#include "arcane/CaseOptionBase.h"
#include "arcane/CaseOptions.h"
#include "arcane/CaseOptionsMulti.h"
#include "arcane/CaseOptionService.h"
#include "arcane/StandardCaseFunction.h"
#include "arcane/ICaseOptions.h"
#include "arcane/ICaseMng.h"
#include "arcane/ICaseFunctionProvider.h"
#include "arcane/ApplicationBuildInfo.h"

#include "arcane/MeshVariableRef.h"
#include "arcane/PrivateVariableScalar.h"
#include "arcane/MeshVariableScalarRef.h"
#include "arcane/MeshVariableArrayRef.h"
#include "arcane/MeshPartialVariableScalarRef.h"
#include "arcane/MeshPartInfo.h"
#include "arcane/MeshKind.h"
#include "arcane/IMeshUtilities.h"

#include "arcane/ServiceInfo.h"

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/StdNum.h"
#include "arcane/datatype/ScalarVariant.h"
#include "arcane/datatype/ArrayVariant.h"
#include "arcane/datatype/DataStorageTypeInfo.h"
#include "arcane/datatype/DataAllocationInfo.h"

#include "arcane/impl/ArcaneMain.h"
#include "arcane/impl/ArcaneSimpleExecutor.h"
#include "arcane/std/ArcaneStdRegisterer.h"

#include "arcane/XmlNode.h"
#include "arcane/XmlNodeList.h"
#include "arcane/XmlNodeIterator.h"

#include "arcane/Observer.h"
#include "arcane/IObservable.h"

#include "arcane/ITimeHistoryMng.h"

#include "arcane/IUnitTest.h"
#include "arcane/BasicUnitTest.h"

#include "arcane/ISerializedData.h"
#include "arcane/IDataReader.h"
#include "arcane/IDataWriter.h"
#include "arcane/IVariableReader.h"
#include "arcane/IDirectExecution.h"
#include "arcane/ICheckpointReader.h"
#include "arcane/ICheckpointWriter.h"
#include "arcane/ITimeHistoryCurveWriter.h"
#include "arcane/ITimeHistoryCurveWriter2.h"
#include "arcane/IMeshReader.h"

#include "arcane/SharedVariable.h"
#include "arcane/ArcaneException.h"

#include "arcane/AbstractDataVisitor.h"
#include "arcane/IMainFactory.h"
#include "arcane/MeshReaderMng.h"
#include "arcane/DotNetRuntimeInitialisationInfo.h"
#include "arcane/utils/ExternalRef.h"
#include "arcane/accelerator/core/AcceleratorRuntimeInitialisationInfo.h"
#include "arcane/accelerator/core/DeviceId.h"

#include "ArcaneSwigUtils.h"

using namespace Arcane;
