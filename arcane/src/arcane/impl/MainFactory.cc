// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MainFactory.cc                                              (C) 2000-2025 */
/*                                                                           */
/* AbstractFactory des gestionnaires d'Arcane.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/MainFactory.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/IArcaneMain.h"
#include "arcane/ISubDomain.h"
#include "arcane/IModuleMng.h"
#include "arcane/IModule.h"
#include "arcane/IModuleMaster.h"
#include "arcane/ServiceUtils.h"
#include "arcane/IFactoryService.h"
#include "arcane/IMeshFactory.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/IApplication.h"
#include "arcane/ItemGroup.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/IMeshMng.h"
#include "arcane/MeshHandle.h"
#include "arcane/MeshBuildInfo.h"

#include "arcane/impl/internal/MeshFactoryMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT IArcaneMain*
createArcaneMainBatch(const ApplicationInfo& app_info,IMainFactory* main_factory);
extern "C++" IApplication* arcaneCreateApplication(IArcaneMain*);
extern "C++" IVariableMng* arcaneCreateVariableMng(ISubDomain*);
extern "C++" IModuleMng* arcaneCreateModuleMng(ISubDomain*);
extern "C++" IEntryPointMng* arcaneCreateEntryPointMng(ISubDomain*);
extern "C++" ARCANE_IMPL_EXPORT ITimeHistoryMng* arcaneCreateTimeHistoryMng2(ISubDomain*);
extern "C++" ICaseMng* arcaneCreateCaseMng(ISubDomain*);
extern "C++" ICaseDocument* arcaneCreateCaseDocument(ITraceMng*,const String& lang);
extern "C++" ICaseDocument* arcaneCreateCaseDocument(ITraceMng*,IXmlDocumentHolder* doc);
extern "C++" ITimeStats* arcaneCreateTimeStats(ITimerMng* timer_mng,ITraceMng* trm,const String& name);
extern "C++" ITimeLoopMng* arcaneCreateTimeLoopMng(ISubDomain*);
extern "C++" IServiceLoader* arcaneCreateServiceLoader();
extern "C++" IServiceMng* arcaneCreateServiceMng(IBase*);
extern "C++" ICheckpointMng* arcaneCreateCheckpointMng(ISubDomain*);
extern "C++" IPropertyMng* arcaneCreatePropertyMng(ITraceMng*);
extern "C++" Ref<IPropertyMng> arcaneCreatePropertyMngReference(ITraceMng*);
extern "C++" IDataFactory* arcaneCreateDataFactory(IApplication*);
extern "C++" Ref<IDataFactoryMng> arcaneCreateDataFactoryMngRef(IApplication*);
extern "C++" ITraceMng* arcaneCreateTraceMng();
extern "C++" ITraceMngPolicy* arcaneCreateTraceMngPolicy(IApplication*);
extern "C++" ILoadBalanceMng* arcaneCreateLoadBalanceMng(ISubDomain*);

extern "C++" ARCANE_CORE_EXPORT IModuleMaster*
arcaneCreateModuleMaster(ISubDomain*);

extern "C++" ARCANE_CORE_EXPORT ITimeLoop*
arcaneCreateTimeLoop(IApplication* sm,const String& name);

extern "C++" ARCANE_IMPL_EXPORT IIOMng*
arcaneCreateIOMng(IParallelSuperMng*);

extern "C++" ARCANE_IMPL_EXPORT IIOMng*
arcaneCreateIOMng(IParallelMng*);

namespace Accelerator
{
extern "C++" ARCANE_IMPORT Ref<IAcceleratorMng>
arccoreCreateAcceleratorMngRef(ITraceMng* tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
//! Renvoie le nom de la fabrique du maillage.
String _getMeshFactoryName(bool is_amr)
{
  if (is_amr)
    return String("ArcaneDynamicAMRMeshFactory");
  return String("ArcaneDynamicMeshFactory");
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MainFactory::
MainFactory()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MainFactory::
~MainFactory()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IArcaneMain* MainFactory::
createArcaneMain(const ApplicationInfo& app_info)
{ 
  return createArcaneMainBatch(app_info,this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IApplication* MainFactory::
createApplication(IArcaneMain* am)
{
  return arcaneCreateApplication(am);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableMng* MainFactory::
createVariableMng(ISubDomain* sd)
{
  return arcaneCreateVariableMng(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IModuleMng*  MainFactory::
createModuleMng(ISubDomain* sd)
{
  return arcaneCreateModuleMng(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IEntryPointMng* MainFactory::
createEntryPointMng(ISubDomain* sd)
{
  return arcaneCreateEntryPointMng(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITimeHistoryMng* MainFactory::
createTimeHistoryMng(ISubDomain* sd)
{
  ITimeHistoryMng* thm = 0;
  thm = arcaneCreateTimeHistoryMng2(sd);
  return thm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseMng* MainFactory::
createCaseMng(ISubDomain* sd)
{
  return arcaneCreateCaseMng(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITimeStats* MainFactory::
createTimeStats(ISubDomain* sd)
{
  ITimerMng* tm = sd->timerMng();
  String name = String::format("Rank{0}", sd->subDomainId());
  return arcaneCreateTimeStats(tm,sd->traceMng(),name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITimeStats* MainFactory::
createTimeStats(ITimerMng* tim,ITraceMng* trm,const String& name)
{
  return arcaneCreateTimeStats(tim,trm,name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITimeLoopMng* MainFactory::
createTimeLoopMng(ISubDomain* sd)
{
  return arcaneCreateTimeLoopMng(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITimeLoop* MainFactory::
createTimeLoop(IApplication* sm,const String& name)
{
  return arcaneCreateTimeLoop(sm,name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IIOMng* MainFactory::
createIOMng(IApplication* app)
{
  return arcaneCreateIOMng(app->parallelSuperMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IIOMng* MainFactory::
createIOMng(IParallelMng* pm)
{
  return arcaneCreateIOMng(pm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IServiceLoader* MainFactory::
createServiceLoader()
{
  return arcaneCreateServiceLoader();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IServiceMng* MainFactory::
createServiceMng(IBase* base)
{
  return arcaneCreateServiceMng(base);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICheckpointMng* MainFactory::
createCheckpointMng(ISubDomain* sd)
{
  return arcaneCreateCheckpointMng(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPropertyMng* MainFactory::
createPropertyMng(ISubDomain* sd)
{
  return arcaneCreatePropertyMng(sd->traceMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IPropertyMng> MainFactory::
createPropertyMngReference(ISubDomain* sd)
{
  return arcaneCreatePropertyMngReference(sd->traceMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseDocument* MainFactory::
createCaseDocument(IApplication* sm)
{
  return arcaneCreateCaseDocument(sm->traceMng(),String());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseDocument* MainFactory::
createCaseDocument(IApplication* sm,const String& lang)
{
  return arcaneCreateCaseDocument(sm->traceMng(),lang);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseDocument* MainFactory::
createCaseDocument(IApplication* sm,IXmlDocumentHolder* doc)
{
  return arcaneCreateCaseDocument(sm->traceMng(),doc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MainFactory::
createMesh(ISubDomain* sd,const String& name, eMeshAMRKind amr_type)
{
  return createMesh(sd,sd->parallelMng(),name, amr_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MainFactory::
createMesh(ISubDomain* sd,const String& name, bool is_amr)
{
  return createMesh(sd,sd->parallelMng(),name, (is_amr ? eMeshAMRKind::Cell : eMeshAMRKind::None));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MainFactory::
createMesh(ISubDomain* sd,IParallelMng* pm,const String& name, bool is_amr)
{
  return createMesh(sd, pm, name, (is_amr ? eMeshAMRKind::Cell : eMeshAMRKind::None));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MainFactory::
createMesh(ISubDomain* sd,IParallelMng* pm,const String& name, eMeshAMRKind amr_type)
{
  String factory_name = _getMeshFactoryName(amr_type != eMeshAMRKind::None);
  IMeshFactoryMng* mfm = sd->meshMng()->meshFactoryMng();
  MeshBuildInfo build_info(name);
  MeshKind mk(build_info.meshKind());
  mk.setMeshAMRKind(amr_type);
  build_info.addMeshKind(mk);
  build_info.addFactoryName(factory_name);
  build_info.addParallelMng(makeRef(pm));
  return mfm->createMesh(build_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MainFactory::
createMesh(ISubDomain* sd,const String& name)
{
  return createMesh(sd,sd->parallelMng(),name,eMeshAMRKind::None);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MainFactory::
createMesh(ISubDomain* sd,IParallelMng* pm,const String& name)
{
  return createMesh(sd,pm,name,eMeshAMRKind::None);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* MainFactory::
createSubMesh(IMesh* mesh,const ItemGroup& group,const String& name)
{
  // Actuellement, les sous-maillages des maillages AMR ne sont pas supportés
  bool is_amr = false;
  String factory_name = _getMeshFactoryName(is_amr);
  IMeshFactoryMng* mfm = mesh->meshMng()->meshFactoryMng();
  MeshBuildInfo build_info(name);
  build_info.addFactoryName(factory_name);
  build_info.addParentGroup(group);
  return mfm->createMesh(build_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDataFactory* MainFactory::
createDataFactory(IApplication* sm)
{
  return arcaneCreateDataFactory(sm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IDataFactoryMng> MainFactory::
createDataFactoryMngRef(IApplication* app)
{
  return arcaneCreateDataFactoryMngRef(app);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IAcceleratorMng> MainFactory::
createAcceleratorMngRef(ITraceMng* tm)
{
  return Accelerator::arccoreCreateAcceleratorMngRef(tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* MainFactory::
createTraceMng()
{
  return arcaneCreateTraceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMngPolicy* MainFactory::
createTraceMngPolicy(IApplication* app)
{
  return arcaneCreateTraceMngPolicy(app);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IModuleMaster* MainFactory::
createModuleMaster(ISubDomain* sd)
{
  IModuleMaster* mm = arcaneCreateModuleMaster(sd);
  IModule* m = dynamic_cast<IModule*>(mm);
  if (!m)
    ARCANE_FATAL("module is not a derived class of 'IModule'");
  sd->moduleMng()->addModule(makeRef(m));
  return mm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ILoadBalanceMng* MainFactory::
createLoadBalanceMng(ISubDomain* sd)
{
  return arcaneCreateLoadBalanceMng(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
