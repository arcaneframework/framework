// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCeaVerifierModule.cc                                  (C) 2000-2016 */
/*                                                                           */
/* Module de vérification.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/EntryPoint.h"
#include "arcane/ISubDomain.h"
#include "arcane/ModuleFactory.h"
#include "arcane/IVerifierService.h"
#include "arcane/ServiceUtils.h"
#include "arcane/IVariableMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/ObserverPool.h"
#include "arcane/ICheckpointMng.h"
#include "arcane/MeshVisitor.h"

#include "arcane/cea/ArcaneCeaVerifier_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de vérification.
 */
class ArcaneCeaVerifierModule
: public ArcaneArcaneCeaVerifierObject
{
 public:

  ArcaneCeaVerifierModule(const ModuleBuildInfo& mb);

 public:

  virtual VersionInfo versionInfo() const { return VersionInfo(0,0,1); }

 public:

  virtual void onExit();
  virtual void onInit();

 private:

  ObserverPool m_observers;

 private:

  void _checkSortedGroups();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_ARCANECEAVERIFIER(ArcaneCeaVerifierModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCeaVerifierModule::
ArcaneCeaVerifierModule(const ModuleBuildInfo& mb)
: ArcaneArcaneCeaVerifierObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCeaVerifierModule::
onInit()
{
  IVariableMng* vm = subDomain()->variableMng();
  IMesh* mesh = defaultMesh();
  Int64UniqueArray uids;
  Int32UniqueArray local_ids;
  const CaseOptionsArcaneCeaVerifier::CaseOptionTrace& traces(options()->trace);
  for( Integer i=0; i<traces.size(); ++i ){
    String var_name(traces[i]->variableName());
    IVariable* var = vm->findVariable(var_name);
    if (!var){
      warning() << "La variable '" << var_name
                << "' demandé en trace n'existe pas";
      continue;
    }
    if (!var->isUsed()){
      warning() << "La variable '" << var_name
                << "' demandé en trace n'est pas utilisée";
      continue;
    }
    eItemKind ik = var->itemKind();
    if (ik==IK_Unknown){
      warning() << "La variable '" << var_name
                << "' demandé en trace n'est pas une variable"
                << " associée à une entité de maillage";
      continue;
    }

    IntegerConstArrayView iuids(traces[i]->uniqueId);
    Integer nb_uid= iuids.size();
    uids.resize(nb_uid);
    local_ids.resize(nb_uid);
    for( Integer z=0; z<nb_uid; ++z )
      uids[z] = iuids[z];
    IItemFamily* family = mesh->itemFamily(ik);
    family->itemsUniqueIdToLocalId(local_ids,uids,false);
    for( Integer z=0; z<nb_uid; ++z ){
      info() << "Trace l'entité uid=" << local_ids[z] << " de '" << var_name << "'";
      var->setTraceInfo(local_ids[z],TT_Read);
    }
    var->syncReferences();
  }
  if (options()->verify() || options()->generate()){
    //subDomain()->timeLoopMng()->setVerificationActive(true);
  }

  if (!platform::getEnvironmentVariable("ARCANE_VERIFY_SORTEDGROUP").null()){
    info() << "Add observer to check sorted groups before checkpoint";
    // Ajoute un observer signalant les groupes non triés.
    m_observers.addObserver(this,
                            &ArcaneCeaVerifierModule::_checkSortedGroups,
                            subDomain()->checkpointMng()->writeObservable());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCeaVerifierModule::
_checkSortedGroups()
{
  // Affiche la liste des groupes non triés.
  Integer nb_sorted_group = 0;
  Integer nb_group = 0;
  auto check_sorted_func = [&](ItemGroup& g)
  {
    ++nb_group;
    if (g.checkIsSorted())
      ++nb_sorted_group;
    else
      info() << "VerifierModule: group not sorted name=" << g.name();
  };
  meshvisitor::visitGroups(mesh(),check_sorted_func);
  info() << "VerifierModule: nb_unsorted_group=" << (nb_group-nb_sorted_group)
         << "/" << nb_group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCeaVerifierModule::
onExit()
{
 if (!platform::getEnvironmentVariable("ARCANE_NO_VERIFY").null())
    return;
  if (!options()->verify())
    return;

  IParallelMng* pm = subDomain()->parallelMng();
  bool is_parallel = pm->isParallel();
  Integer sid = pm->commRank();

  ServiceBuilder<IVerifierService> sf(subDomain());
  auto service2(sf.createReference("ArcaneBasicVerifier2"));

  bool compare_from_sequential = options()->compareParallelSequential();
  if (!is_parallel)
    compare_from_sequential = false;

  String result_file_name = options()->resultFile();
  if (result_file_name.null())
    result_file_name = "compare.xml";
  service2->setResultFileName(result_file_name);

  String reference_file_name = options()->referenceFile();
  if (reference_file_name.null())
    reference_file_name = "verif";
  String base_reference_file_name = reference_file_name;
  if (is_parallel && !compare_from_sequential){
    reference_file_name = reference_file_name + "." + sid;
  }
  service2->setFileName(base_reference_file_name+"2");

  // Nouvelle version uniquement avec le BasicVerifier et plus HDF5.
  if (options()->generate()){
    info() << "Ecriture du fichier de vérification";
    service2->writeReferenceFile();
  }
  else{
    service2->doVerifFromReferenceFile(compare_from_sequential,false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
