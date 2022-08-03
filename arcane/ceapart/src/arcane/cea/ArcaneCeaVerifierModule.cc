// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneVerifierModule.cc                                     (C) 2000-2022 */
/*                                                                           */
/* Module de vérification.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de vérification.
 */
class ArcaneVerifierModule
: public ArcaneArcaneCeaVerifierObject
{
 public:

  explicit ArcaneVerifierModule(const ModuleBuildInfo& mb);

 public:

  VersionInfo versionInfo() const override { return VersionInfo(0,0,1); }

 public:

  void onExit() override;
  void onInit() override;

 private:

  ObserverPool m_observers;

 private:

  void _checkSortedGroups();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneVerifierModule::
ArcaneVerifierModule(const ModuleBuildInfo& mb)
: ArcaneArcaneCeaVerifierObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneVerifierModule::
onInit()
{
  if (options()->trace.size()!=0)
    this->pwarning() << "The option 'trace' is no longer used and should be removed";

  if (!platform::getEnvironmentVariable("ARCANE_VERIFY_SORTEDGROUP").null()){
    info() << "Add observer to check sorted groups before checkpoint";
    // Ajoute un observer signalant les groupes non triés.
    m_observers.addObserver(this,
                            &ArcaneVerifierModule::_checkSortedGroups,
                            subDomain()->checkpointMng()->writeObservable());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneVerifierModule::
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
  meshvisitor::visitGroups(this->mesh(),check_sorted_func);
  info() << "VerifierModule: nb_unsorted_group=" << (nb_group-nb_sorted_group)
         << "/" << nb_group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneVerifierModule::
onExit()
{
  if (!options()->verify())
    return;
  if (!platform::getEnvironmentVariable("ARCANE_NO_VERIFY").null())
    return;

  IParallelMng* pm = subDomain()->parallelMng();
  bool is_parallel = pm->isParallel();
  Int32 rank = pm->commRank();

  Ref<IVerifierService> verifier_service;

  {
    ServiceBuilder<IVerifierService> sf(subDomain());

    String verifier_service_name = options()->verifierServiceName();
    // Autorise pour test une variable d'environnement pour surcharger le service
    String env_service_name = platform::getEnvironmentVariable("ARCANE_VERIFIER_SERVICE");
    if (!env_service_name.null())
      verifier_service_name = env_service_name;
    info() << "Verification Module using service=" << verifier_service_name;
    verifier_service = sf.createReference(verifier_service_name);
  }

  bool compare_from_sequential = options()->compareParallelSequential();
  if (!is_parallel)
    compare_from_sequential = false;

  String result_file_name = options()->resultFile();
  if (result_file_name.empty())
    result_file_name = "compare.xml";
  verifier_service->setResultFileName(result_file_name);

  String reference_file_name = options()->referenceFile();
  if (reference_file_name.empty())
    reference_file_name = "check";

  info() << "Verification check is_parallel?=" << is_parallel << " compare_from_sequential?=" << compare_from_sequential;
  String base_file_name = reference_file_name;
  verifier_service->setFileName(base_file_name);

  if (options()->generate()){
    info() << "Writing check file '" << base_file_name << "'";
    verifier_service->writeReferenceFile();
  }
  else{
    info() << "Comparing reference file '" << base_file_name << "'";
    verifier_service->doVerifFromReferenceFile(compare_from_sequential,false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_ARCANECEAVERIFIER(ArcaneVerifierModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

