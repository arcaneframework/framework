// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneVerifierModule.cc                                     (C) 2000-2010 */
/*                                                                           */
/* Module de vérification.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/EntryPoint.h"
#include "arcane/ISubDomain.h"
#include "arcane/ModuleFactory.h"
#include "arcane/IVerifierService.h"
#include "arcane/ServiceUtils.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/IVariableMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ITimeLoopMng.h"

#include "arcane/std/ArcaneVerifier_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de vérification.
 */
class ArcaneVerifierModule
: public ArcaneArcaneVerifierObject
{
 public:

  ArcaneVerifierModule(const ModuleBuilder& mb);

 public:

  virtual VersionInfo versionInfo() const { return VersionInfo(0,0,1); }

 public:

  virtual void onExit();
  virtual void onInit();

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_ARCANEVERIFIER(ArcaneVerifierModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneVerifierModule::
ArcaneVerifierModule(const ModuleBuilder& mb)
: ArcaneArcaneVerifierObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneVerifierModule::
onInit()
{
  IVariableMng* vm = subDomain()->variableMng();
  IMesh* mesh = subDomain()->defaultMesh();
  Int64UniqueArray uids;
  Int32UniqueArray local_ids;
  const CaseOptionsArcaneVerifier::CaseOptionTrace& traces(options()->trace);
  for( Integer i=0; i<traces.size(); ++i ){
    String var_name(traces[i]->variableName());
    IVariable* var = vm->findVariable(var_name);
    if (!var){
      warning() << "Variable '" << var_name
                << "' required by trace does not exist";
      continue;
    }
    if (!var->isUsed()){
      warning() << "Variable '" << var_name
                << "' required by trace is not used";
      continue;
    }
    eItemKind ik = var->itemKind();
    if (ik==IK_Unknown){
      warning() << "Variable '" << var_name
                << "' required by trace is not a variable"
                << " on a mesh item";
      continue;
    }

    ConstArrayView<Integer> iuids(traces[i]->uniqueId);
    Integer nb_uid= iuids.size();
    uids.resize(nb_uid);
    local_ids.resize(nb_uid);
    for( Integer z=0; z<nb_uid; ++z )
      uids[z] = static_cast<Integer>(iuids[z]);
    mesh->itemFamily(ik)->itemsUniqueIdToLocalId(local_ids,uids,false);
    for( Integer z=0; z<nb_uid; ++z ){
      info() << "Trace item uid=" << uids[z] << " de '" << var_name << "'";
      var->setTraceInfo(local_ids[z],TT_Read);
    }
    var->syncReferences();
  }
  if (options()->verify() || options()->generate()){
    //subDomain()->timeLoopMng()->setVerificationActive(true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneVerifierModule::
onExit()
{
  if (!options()->verify())
    return;

  IParallelMng* pm = subDomain()->parallelMng();
  bool is_parallel = pm->isParallel();
  Integer sid = pm->commRank();

  String service_name("ArcaneBasicVerifier2");
  ServiceBuilder<IVerifierService> sf(subDomain());
  auto service(sf.createReference(service_name));

  bool compare_from_sequential = options()->compareParallelSequential();
  if (!is_parallel)
    compare_from_sequential = false;

  String result_file_name = options()->resultFile();
  if (result_file_name.null())
    result_file_name = "compare.xml";
  service->setResultFileName(result_file_name);

  String reference_file_name = options()->referenceFile();
  if (reference_file_name.null())
    reference_file_name = "check";
  if (is_parallel && !compare_from_sequential){
    reference_file_name = reference_file_name + "." + sid;
  }
  service->setFileName(reference_file_name);

  //const CommonVariables& vc = subDomain()->commonVariables();
  if (options()->generate()){
    info() << "Writing check file";
    service->writeReferenceFile();
  }
  else{
    service->doVerifFromReferenceFile(compare_from_sequential,false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
