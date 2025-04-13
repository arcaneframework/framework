// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeLoopMng.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de la boucle en temps.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iterator.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/IMemoryInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/IProfilingService.h"
#include "arcane/utils/IMessagePassingProfilingService.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/TimeoutException.h"
#include "arcane/utils/GoBackwardException.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/FloatingPointExceptionSentry.h"
#include "arcane/utils/JSONWriter.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/IServiceLoader.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IEntryPoint.h"
#include "arcane/core/IEntryPointMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IModule.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IModuleMng.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/TimeLoopSingletonServiceInfo.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/ICaseFunction.h"
#include "arcane/core/IServiceFactory.h"
#include "arcane/core/IModuleFactory.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/ServiceInfo.h"
#include "arcane/core/ServiceUtils.h"
#include "arcane/core/CaseOptions.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/IVerifierService.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/IVariableFilter.h"
#include "arcane/core/ITimeStats.h"
#include "arcane/core/XmlNodeIterator.h"
#include "arcane/core/ServiceFinder2.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/Observable.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IConfiguration.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IVariableUtilities.h"
#include "arcane/core/IItemEnumeratorTracer.h"
#include "arcane/core/ObservablePool.h"
#include "arcane/core/parallel/IStat.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/IVariableSynchronizerMng.h"
#include "arcane/core/VariableComparer.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/Runner.h"

#include "arcane/impl/DefaultBackwardMng.h"

#include <algorithm>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de la boucle en temps.
 */
class TimeLoopMng
: public TraceAccessor
, public ITimeLoopMng
{
 public:

  struct ModuleState
  {
    ModuleState(bool is_optional, const String & alias)
    : m_is_active(false), m_is_optional(is_optional), m_alias(alias) { }
    
    bool m_is_active;
    bool m_is_optional;
    String m_alias;
  };
  enum eVerifType
  {
    VerifNone, //!< Indique qu'on ne fait pas de vérifications
    VerifWrite, //!< Indique qu'on génère des valeurs pour vérifier
    VerifRead, //!< Indique qu'on relit et vérifie des valeurs
    VerifSync, //! Vérifie que les variables sont synchronisées
    VerifSameReplica //! Vérifie que les variables ont les mêmes valeurs sur tous les réplica.
  };

  //! Liste des états des modules
  typedef std::map<String,ModuleState> ModuleStateMap;
  //! Liste de boucles en temps
  typedef std::map<String,ITimeLoop*> TimeLoopMap;
  //! Liste des fabriques des modules indéxés par leur nom.
  typedef std::map<String,IModuleFactoryInfo*> ModuleFactoryMap;

 public:

  explicit TimeLoopMng(ISubDomain* sd);
  ~TimeLoopMng() override;

 public:

  void build() override;

  ISubDomain* subDomain() const override { return m_sub_domain; }

  void execExitEntryPoints() override;
  void execBuildEntryPoints() override;
  void execInitEntryPoints(bool is_continue) override;
  void stopComputeLoop(bool is_final_time,bool has_error) override;
  bool finalTimeReached() const override { return m_final_time_reached; }
  Real cpuTimeUsed() const override;

  EntryPointCollection loopEntryPoints() override { return m_loop_entry_points; }
  EntryPointCollection usedTimeLoopEntryPoints() override { return m_used_time_loop_entry_points; }

  void registerTimeLoop(ITimeLoop* timeloop) override;
  void setUsedTimeLoop(const String& name) override;

  ITimeLoop* usedTimeLoop() const override { return m_used_time_loop; }

  void doExecNextEntryPoint(bool & is_last) override;
  IEntryPoint* nextEntryPoint() override;

  IEntryPoint* currentEntryPoint() override { return m_current_entry_point_ptr; }

  int doOneIteration() override;

  void setBackwardMng(IBackwardMng* backward_mng) override;
  // Attention, peut renvoyer NULL
  IBackwardMng* getBackwardMng() const override { return m_backward_mng;}

  void goBackward() override;
  bool isDoingBackward() override;
  void setBackwardSavePeriod(Integer n) override
  {
    if (!m_backward_mng)
      _createOwnDefaultBackwardMng();
    m_backward_mng->setSavePeriod(n);
  }

  void setVerificationActive(bool is_active) override { m_verification_active = is_active; }
  void doVerification(const String& name) override;

  void registerActionMeshPartition(IMeshPartitionerBase* mesh_partitioner) override;

  void timeLoopsName(StringCollection & names) const override;
  void timeLoops(TimeLoopCollection & time_loops) const override;
  ITimeLoop * createTimeLoop(const String & name) override;

  int doComputeLoop(Integer max_loop) override;
  Integer nbLoop() const override { return m_nb_loop; }

  void setStopReason(eTimeLoopStopReason reason) override;
  eTimeLoopStopReason stopReason() const override { return m_stop_reason; }

 public:

  void execRestoreEntryPoints();
  void execOnMeshChangedEntryPoints() override;
  void execOnMeshRefinementEntryPoints() override;

  ITraceMng* traceMng() { return m_sub_domain->traceMng(); }

  IObservable* observable(eTimeLoopEventType type) override
  {
    return m_observables[type];
  }

 protected:

  //! Ajoute un point d'entrée à exécuter
  void _addExecuteEntryPoint(IEntryPoint*);
  //! Crée un module à partir de son nom
  bool _createModule(const String & module_name);
  //! Ajoute à la liste \a entry_points les points d'entrée proposés dans \a entry_points_info
  /*! Controle que ceux ci sont bien associé à \a where */
  void _processEntryPoints(EntryPointList& entry_points,
                           const TimeLoopEntryPointInfoCollection& entry_points_info,
                           const char* where);
  /**
   * Renseigne la liste d'état des modules en fonction des informations
   * contenues dans les éléments <modules> de la boucle en temps et du fichier de données.
   */
  void _fillModuleStateMap(ITimeLoop* time_loop);

  /*!
   * Retourne le nom du module et le nom du point d'entrée à partir du nom
   * référencé dans la boucle en temps, de type ModuleName.EntryPointName.
   */
  static void _extractModuleAndEntryPointName(const String & timeloop_call_name,
                                              String& module_name,String& entry_point_name);

 private:

  //! Gestionnaire du sous-domaine.
  ISubDomain* m_sub_domain;

  //! Gestionnaire de points d'entrée
  IEntryPointMng* m_entry_point_mng;

  //! Liste des modules à éxécuter
  ModuleList m_list_execute_module;

  //! Liste des points d'entrée à exécuter lors de la construction
  EntryPointList m_build_entry_points;

  //! Liste des points d'entrée à exécuter
  EntryPointList m_loop_entry_points;

  //! Liste des points d'entrée à exécuter à l'initialisation
  EntryPointList m_init_entry_points;

  //! Liste des points d'entrée à exécuter à la terminaison
  EntryPointList m_exit_entry_points;

  //! Liste des points d'entrée à exécuter lors d'un retour arrière
  EntryPointList m_restore_entry_points;

  //! Liste des points d'entrée à exécuter après un changement de maillage
  EntryPointList m_on_mesh_changed_entry_points;

  //! Liste des points d'entrée à exécuter après un raffinement
  EntryPointList m_on_mesh_refinement_entry_points;

  //! Liste de tous les points d'entrée de la boucle en temps utilisée.
  EntryPointList m_used_time_loop_entry_points;

  //! Liste des boucles en temps
  TimeLoopMap m_time_loop_list;

  ITimeLoop* m_default_time_loop; //!< Boucle en temps par défaut

  ITimeLoop* m_used_time_loop; //!< Boucle en temps utilisée

  IEntryPoint* m_current_entry_point_ptr; //!< Point d'entrée en cours d'exécution

  bool m_stop_time_loop;
  bool m_stop_has_error;
  bool m_final_time_reached;

  Integer m_current_entry_point; //!< Prochain point d'entrée à exécuter

  eVerifType m_verif_type; //!< Type de vérifications
  bool m_verif_same_parallel;
  String m_verif_path; //!< Répertoire de sauvegarde/lecture des verifs
  bool m_verification_active;
  //! Si vrai, effectue vérifications à chaque point d'entrée, sinon uniquement en fin d'itération.
  bool m_verification_at_entry_point;
  bool m_verification_only_at_exit = false;
  eVariableComparerComputeDifferenceMethod m_compute_diff_method = eVariableComparerComputeDifferenceMethod::Relative;

  IBackwardMng* m_backward_mng; //!< Gestionnaire du retour-arrière;
  bool m_my_own_backward_mng;

  ModuleStateMap m_module_state_list; //! Etat de tous les modules référencés

  ModuleFactoryMap m_module_factory_map; //! Liste des fabriques des modules.
  ModuleFactoryMap m_lang_module_factory_map; //! Liste des fabriques des modules dans la langue du JDD.

  Ref<IVerifierService> m_verifier_service;
  UniqueArray<IMeshPartitionerBase*> m_mesh_partitioner;
  String m_message_class_name;
  Integer m_alarm_timer_value;
  Integer m_nb_loop;
  
  ObservablePool<eTimeLoopEventType> m_observables;

  eTimeLoopStopReason m_stop_reason;

  // Service de message passing profiling
  Ref<IMessagePassingProfilingService> m_msg_pass_prof_srv;

  //! Pour test, point d'entrée spécifique à appeler
  String m_specific_entry_point_name;

 private:

  void _execOneEntryPoint(IEntryPoint* ic, Integer index_value = 0, bool do_verif = false);
  void _dumpTimeInfos(JSONWriter& json_writer);
  void _resetTimer() const;
  void _checkVerif(const String& entry_point_name,Integer index,bool do_verif);
  void _checkVerifSameOnAllReplica(const String& entry_point_name);
  void _createOwnDefaultBackwardMng();
  void _doMeshPartition();
  void _fillModuleFactoryMap();
  void _createSingletonServices(IServiceLoader* service_loader);
  void _callSpecificEntryPoint();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ITimeLoopMng*
arcaneCreateTimeLoopMng(ISubDomain * mng)
{
  auto* tm = new TimeLoopMng(mng);
  return tm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeLoopMng::
TimeLoopMng(ISubDomain* sd)
: TraceAccessor(sd->traceMng())
, m_sub_domain(sd)
, m_entry_point_mng(m_sub_domain->entryPointMng())
, m_default_time_loop(nullptr)
, m_used_time_loop(nullptr)
, m_current_entry_point_ptr(nullptr)
, m_stop_time_loop(false)
, m_stop_has_error(false)
, m_final_time_reached(false)
, m_current_entry_point(0)
, m_verif_type(VerifNone)
, m_verif_same_parallel(false)
, m_verif_path(".")
, m_verification_active(false)
, m_verification_at_entry_point(false)
, m_backward_mng(nullptr)
, m_my_own_backward_mng(false)
, m_message_class_name("TimeLoopMng")
, m_alarm_timer_value(0)
, m_nb_loop(0)
, m_stop_reason(eTimeLoopStopReason::NoStop)
{
  {
    String s = platform::getEnvironmentVariable("ARCANE_LISTENER_TIMEOUT");
    if (!s.null()){
      Integer v = 0;
      if (!builtInGetValue(v,s))
        if (v>0)
          m_alarm_timer_value = v;        
    }
  }
  {
    String s = platform::getEnvironmentVariable("ARCANE_VERIF_PARALLEL");
    if (!s.null())
      m_verif_same_parallel = true;
  }

  m_observables.add(eTimeLoopEventType::BeginEntryPoint);
  m_observables.add(eTimeLoopEventType::EndEntryPoint);
  m_observables.add(eTimeLoopEventType::BeginIteration);
  m_observables.add(eTimeLoopEventType::EndIteration);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeLoopMng::
~TimeLoopMng()
{
  for( ConstIterT<TimeLoopMap> i(m_time_loop_list); i(); ++i){
    ITimeLoop * tm = i->second;
    delete tm;
  }

  if (m_my_own_backward_mng)
    delete m_backward_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
build()
{
  // Créé en enregistre une boucle par défaut.
  ITimeLoop * tm = createTimeLoop(String("ArcaneEmptyLoop"));
  m_default_time_loop = tm;
  registerTimeLoop(tm);

  {
    String verif_env = platform::getEnvironmentVariable("STDENV_VERIF");
    if (verif_env=="READ"){
      m_verif_type = VerifRead;
      info() << "Checking in read mode";
    }
    if (verif_env=="WRITE"){
      m_verif_type = VerifWrite;
      info() << "Checking in write mode";
    }
    if (verif_env=="CHECKSYNC"){
      m_verif_type = VerifSync;
      info() << "Checking synchronizations";
    }
    if (verif_env=="CHECKREPLICA"){
      m_verif_type = VerifSameReplica;
      info() << "Checking variables values between replica";
    }
    if (m_verif_type!=VerifNone)
      m_verification_active = true;
  }

  {
    String s = platform::getEnvironmentVariable("STDENV_VERIF_DIFF_METHOD");
    if (!s.null()){
      if (s=="RELATIVE"){
        m_compute_diff_method = eVariableComparerComputeDifferenceMethod::Relative;
        info() << "Using 'Relative' method to compute difference of variable values";
      }
      if (s=="LOCALNORMMAX"){
        m_compute_diff_method = eVariableComparerComputeDifferenceMethod::LocalNormMax;
        info() << "Using 'LocalNormMax' method to compute difference of variable values";
      }
    }
  }
  {
    String s = platform::getEnvironmentVariable("STDENV_VERIF_ENTRYPOINT");
    if (!s.null()){
      m_verification_at_entry_point = true;
      info() << "Do verification at each entry point";
    }
  }
  {
    String s = platform::getEnvironmentVariable("STDENV_VERIF_ONLY_AT_EXIT");
    if (s=="1" || s=="true" || s=="TRUE"){
      m_verification_only_at_exit = true;
      info() << "Do verification only at exit";
    }
  }
  // Regarde si on n'exécute qu'un seul point d'entrée au lieu de la boucle
  // en temps. Cela est utilisé uniquement pour des tests
  {
    String s = platform::getEnvironmentVariable("ARCANE_CALL_SPECIFIC_ENTRY_POINT");
    if (!s.null()){
      m_specific_entry_point_name = s;
      info() << "Use specific entry point: " << s;
    }
  }

  //
  if (m_verification_active){
    m_verif_path = platform::getEnvironmentVariable("STDENV_VERIF_PATH");
    if (m_verif_path.null()){
      String user_name = subDomain()->application()->userName();
      m_verif_path = "/tmp/" + user_name + "/verif";
      //m_verif_path += user_name;
      //m_verif_path += "/verif";
    }
  }
  if (m_verif_type==VerifWrite){
    if (subDomain()->parallelMng()->isMasterIO()){
      info() << "Creating directory '"<< m_verif_path << "' for checking";
      platform::recursiveCreateDirectory(m_verif_path);
    }
  }

  // Creation du service de "message passing profiling" le cas echeant
  {
    String msg_pass_prof_str = platform::getEnvironmentVariable("ARCANE_MESSAGE_PASSING_PROFILING");
    if (!msg_pass_prof_str.null()) {
      String service_name;
      // TODO: ne pas faire de if mais utiliser directement le nom spécifié par la
      // variable d'environnement.
      if (msg_pass_prof_str == "JSON") {
        service_name = "JsonMessagePassingProfiling";
      } else if (msg_pass_prof_str == "OTF2") {
		    service_name = "Otf2MessagePassingProfiling";
      }
      ServiceBuilder<IMessagePassingProfilingService> srv(this->subDomain());
      m_msg_pass_prof_srv = srv.createReference(service_name ,SB_AllowNull);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_createOwnDefaultBackwardMng()
{
  m_backward_mng = new DefaultBackwardMng(traceMng(),subDomain());
  m_backward_mng->init();
  m_my_own_backward_mng = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
setBackwardMng(IBackwardMng* backward_mng)
{
  ARCANE_ASSERT((backward_mng),("IBackwardMng pointer null"));

  if (m_backward_mng){
    // Détruit l'ancien gestionnaire si c'est nous qui l'avons créé.
    if (m_my_own_backward_mng)
      delete m_backward_mng;
    ARCANE_FATAL("Backward manager already set");
  }

  m_backward_mng = backward_mng;
  m_my_own_backward_mng = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
execInitEntryPoints(bool is_continue)
{
  Timer::Action ts_action(m_sub_domain,"InitEntryPoints");
  info() << "-- Executing init entry points";

  String where = (is_continue) ? IEntryPoint::WContinueInit : IEntryPoint::WStartInit;

  for( EntryPointList::Enumerator i(m_init_entry_points); ++i; ){
    IEntryPoint * ic = * i;
    if (ic->where() == IEntryPoint::WInit || ic->where() == where)
      _execOneEntryPoint(ic);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
execBuildEntryPoints()
{
  Timer::Action ts_action(m_sub_domain,"BuildtEntryPoints");
  info() << "-- Executing build entry points";

  for( EntryPointList::Enumerator i(m_build_entry_points); ++i; ){
    IEntryPoint* ic = *i;
    if (ic->where()==IEntryPoint::WBuild)
      _execOneEntryPoint(ic);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
execRestoreEntryPoints()
{
  Timer::Action ts_action(m_sub_domain,"RestoreEntryPoints");
  info() << "-- Executing restore entry points";

  for( EntryPointList::Enumerator i(m_restore_entry_points); ++i; ){
    IEntryPoint * ic = * i;
    _execOneEntryPoint(ic);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
execOnMeshChangedEntryPoints()
{
  Timer::Action ts_action(m_sub_domain,"OnMeshChangedEntryPoints");
  info() << "-- Executing entry points after mesh change";

  for( EntryPointList::Enumerator i(m_on_mesh_changed_entry_points); ++i; ){
    IEntryPoint * ic = * i;
    info() << "Execute: " << ic->name();
    _execOneEntryPoint(ic);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
execOnMeshRefinementEntryPoints()
{
  Timer::Action ts_action(m_sub_domain,"OnMeshRefinementEntryPoints");
  info() << "-- Executing entry points after mesh refinement";

  for( EntryPointList::Enumerator i(m_on_mesh_refinement_entry_points); ++i; ){
    IEntryPoint * ic = * i;
    info() << "Execute: " << ic->name();
    _execOneEntryPoint(ic);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
execExitEntryPoints()
{
  Timer::Action ts_action(m_sub_domain,"ExitEntryPoints");
  info() << "-- Executing terminal entry points";

  for (EntryPointList::Enumerator i(m_exit_entry_points); ++i; ){
    IEntryPoint * ic = * i;
    _execOneEntryPoint(ic);
  }

  // Affiche les statistiques d'exécution
  {
    JSONWriter json_writer(JSONWriter::FormatFlags::None);
    json_writer.beginObject();
    _dumpTimeInfos(json_writer);
    json_writer.endObject();
    traceMng()->plog() << "TimeStats:" << json_writer.getBuffer();
    traceMng()->flush();
  }

  // Affiche le profiling de message passing dans un fichier pour les traces JSON
  if (m_msg_pass_prof_srv.get() && m_msg_pass_prof_srv->implName() == "JsonMessagePassingProfiling") {
    String fullname(subDomain()->listingDirectory().file("message_passing_logs.")
                    + String(std::to_string(subDomain()->subDomainId()))
                    + String(".json"));
    std::ofstream file(fullname.localstr());
    m_msg_pass_prof_srv->printInfos(file);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_execOneEntryPoint(IEntryPoint * ic, Integer index, bool do_verif)
{
  m_current_entry_point_ptr = ic;
  m_observables[eTimeLoopEventType::BeginEntryPoint]->notifyAllObservers();
  ic->executeEntryPoint();
  m_observables[eTimeLoopEventType::EndEntryPoint]->notifyAllObservers();
  m_current_entry_point_ptr = nullptr;
  if (m_verification_at_entry_point && !m_verification_only_at_exit)
    _checkVerif(ic->name(),index,do_verif);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
doVerification(const String& name)
{
  _checkVerif(name,0,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_checkVerif(const String& entry_point_name,Integer index,bool do_verif)
{
  IParallelMng * pm = subDomain()->parallelMng();

  ISubDomain* sd = subDomain();
  IApplication* app = sd->application();
  VariableComparer variable_comparer(traceMng());

  if ((m_verif_type==VerifRead || m_verif_type==VerifWrite) && !m_verifier_service.get()){
    String service_name1 = platform::getEnvironmentVariable("STDENV_VERIF_SERVICE");
    if (service_name1.empty())
      service_name1 = "ArcaneBasicVerifier2";

    ServiceFinder2T<IVerifierService,ISubDomain> sf(app,sd);
    m_verifier_service = sf.createReference(service_name1);
    if (!m_verifier_service.get()){
      warning() << "No verification service is available."
                << " No verification will be performed";
      m_verif_type = VerifNone;
    }
    if (m_verifier_service.get()){
      info() << "Use the service <" << service_name1
             << "> for verification";
    }
  }

  if (pm->isParallel()){
    if (m_verif_type == VerifSync){
      Integer nb_error = 0;
      VariableCollection variables(subDomain()->variableMng()->usedVariables());
      for( VariableCollection::Enumerator i(variables); ++i; ){
        IVariable* var = *i;
        if (var->property() & IVariable::PNoNeedSync)
          continue;
        if (var->isPartial())
          continue;
        nb_error += variable_comparer.checkIfSync(var, 5);
      }
      if (nb_error!=0)
        info() << "Error in synchronization nb_error=" << nb_error
               << " entry_point=" << entry_point_name;
    }
  }

  if (m_verif_type == VerifSameReplica && pm->replication()->hasReplication()){
    _checkVerifSameOnAllReplica(entry_point_name);
  }

  if ((m_verif_type == VerifRead || m_verif_type == VerifWrite) && do_verif){
    Integer current_iter = subDomain()->commonVariables().globalIteration();
    if (current_iter>=0){
      {
        StringBuilder path = Directory(m_verif_path).file("verif_file");
        if (m_verif_same_parallel){
          path += "_";
          path += pm->commRank();
        }

        StringBuilder sub_dir;
        sub_dir += "iter";
        sub_dir += current_iter;
        sub_dir += "/";
        sub_dir += entry_point_name;
        sub_dir += index;
          
        m_verifier_service->setFileName(path.toString());
        m_verifier_service->setSubDir(sub_dir.toString());
        m_verifier_service->setComputeDifferenceMethod(m_compute_diff_method);
      }
      bool parallel_sequential = pm->isParallel();
      if (m_verif_same_parallel)
        parallel_sequential = false;
      if (m_verif_type==VerifRead){

        // Activer cette partie si on souhaite sauver les valeurs actuelles
#if 0
        {
          ServiceFinder2T<IVerifierService,ISubDomain> sf(app,sd);
          ScopedPtrT<IVerifierService> current_save(sf.find(service_name1));
          if (current_save.get()){
            current_save->setFileName(m_verifier_service->fileName()+"_current");
            current_save->setSubDir(m_verifier_service->subDir());
            current_save->writeReferenceFile();
          }
        }
#endif

        // En lecture, désactive les exceptions flottantes pour éviter
        // des erreurs lorsqu'on compare des variables non initialisées
        {
          FloatingPointExceptionSentry fpes(false);
          m_verifier_service->doVerifFromReferenceFile(parallel_sequential,
                                                       platform::getEnvironmentVariable("STDENV_VERIF_SKIP_GHOSTS").null());
        }
      }
      if (m_verif_type==VerifWrite)
        m_verifier_service->writeReferenceFile();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_checkVerifSameOnAllReplica(const String& entry_point_name)
{
  info() << "CHECK: comparing variables values on all replica"
         << " entry_point_name=" << entry_point_name;
  ISubDomain* sd = subDomain();
  IParallelMng* replica_pm = sd->parallelMng()->replication()->replicaParallelMng();
  IVariableMng* vm = sd->variableMng();
  VariableCollection variables(vm->usedVariables());
  VariableList vars_to_check;
  VariableComparer variable_comparer(traceMng());
  for( VariableCollection::Enumerator i(variables); ++i; ){
    IVariable* var = *i;
    if (var->property() & IVariable::PNoReplicaSync)
      continue;
    if (var->isPartial())
      continue;
    // Pour l'instant on ne supporte pas la comparaison entre réplica
    // des variables de type 'String'.
    if (var->dataType()==DT_String)
      continue;
    vars_to_check.add(var);
  }
  VariableCollection common_vars = vm->utilities()->filterCommonVariables(replica_pm,vars_to_check,true);

  Integer nb_error = 0;
  {
    FloatingPointExceptionSentry fpes(false);
    for( VariableCollection::Enumerator ivar(common_vars); ++ivar; ){
      IVariable* var = *ivar;
      nb_error += variable_comparer.checkIfSameOnAllReplica(var, 10);
    }
  }

  if (nb_error!=0)
    info() << "Errors in comparing values between replica nb_error=" << nb_error
           << " entry_point=" << entry_point_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Execute le point d'entrée suivant dans la liste.
 */
void TimeLoopMng::
doExecNextEntryPoint(bool & is_last)
{
  if (!m_used_time_loop)
    ARCANE_FATAL("No time loop");

  is_last = true;

  const EntryPointList & cl = m_loop_entry_points;

  if (cl.empty())
    return;

  if (m_current_entry_point >= cl.count())
    m_current_entry_point = 0;

  _execOneEntryPoint(cl[m_current_entry_point]);

  ++m_current_entry_point;

  is_last = (m_current_entry_point >= cl.count());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne la fonction suivante à appeler.
 */
IEntryPoint* TimeLoopMng::
nextEntryPoint()
{
  //return m_loop_entry_points->EntryPointList()[m_current_entry_point];
  const EntryPointList & cl = m_loop_entry_points;
  if (m_current_entry_point >= cl.count())
    return nullptr;
  return cl[m_current_entry_point];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int TimeLoopMng::
doOneIteration()
{
  ITraceMng * msg = traceMng();
  ISubDomain* sd = subDomain();
  Trace::Setter mci(msg,m_message_class_name);

  if (!m_used_time_loop)
    ARCANE_FATAL("No time loop");

  Integer current_iteration = sd->commonVariables().globalIteration();
  // Le numéro d'itération 0 correspond à l'initialisation. La première
  // itération de la boucle en temps porte le numéro 1.
  if (current_iteration==0){
    VariableScalarInteger global_iteration(sd->commonVariables().m_global_iteration);
    global_iteration = 1;
    current_iteration = 1;
  }

  _resetTimer();
  
  if (m_stop_time_loop){
    if (m_stop_has_error)
      return (-1);
    return (+1);
  }

  m_backward_mng->beginAction();

  // Action de restauration
  if (m_backward_mng->checkAndApplyRestore()) {

    execRestoreEntryPoints();

    // Repartitionnement inutile si retour-arrière
    m_mesh_partitioner.clear();
  }

  // Repartionnement demandé
  bool mesh_partition_done = false;
  if (!m_mesh_partitioner.empty()) {
    _doMeshPartition();
    mesh_partition_done = true;
  }

  // Action de sauvegarde
  m_backward_mng->checkAndApplySave(mesh_partition_done);

  m_backward_mng->endAction();

  {
    // Regarde les modules inactifs au cours du temps
    Real global_time = sd->commonVariables().globalTime();
    CaseOptionsCollection blocks(sd->caseMng()->blocks());
    for( CaseOptionsCollection::Enumerator i(blocks); ++i; ){
      ICaseOptions * opt = * i;
      IModule * mod = opt->caseModule();
      ICaseFunction * f = opt->activateFunction();
      if (mod && f){
        bool is_active = true;
        f->value(global_time, is_active);
        bool mod_disabled = mod->disabled();
        bool mod_new_disabled = !is_active;
        if (mod_new_disabled != mod_disabled){
          if (mod_new_disabled)
            info() << "The module " << mod->name() << " is desactivated";
          else 
            info() << "The module " << mod->name() << " is activated";
          mod->setDisabled(mod_new_disabled);
        }
      }
    }
  }

  m_observables[eTimeLoopEventType::BeginIteration]->notifyAllObservers();

  // Exécute chaque point d'entrée de l'itération
  {
    Integer index =0;
    sd->timeStats()->notifyNewIterationLoop();
    Timer::Action ts_action(sd,"LoopEntryPoints");
    for( EntryPointList::Enumerator i(m_loop_entry_points); ++i; ++index ){
      IEntryPoint* ep = *i;
      IModule* mod = ep->module();
      if (mod && mod->disabled()){
        continue;
        //warning() << "MODULE " << mod->name() << " is disabled";
      }
      try{
        _execOneEntryPoint(*i, index, true);
      } catch(const GoBackwardException&){
        m_backward_mng->goBackward();
      } catch(...){ // On remonte toute autre exception
        throw;
      }
      if (m_backward_mng->isBackwardEnabled()){
        break;
      }
    }
    if (!m_verification_at_entry_point && !m_verification_only_at_exit)
      _checkVerif("_EndLoop",0,true);
  }
  m_observables[eTimeLoopEventType::EndIteration]->notifyAllObservers();

  {
    bool force_prepare_dump = false;
    if (m_verification_active && m_verif_type==VerifWrite){
      if (!m_verif_same_parallel)
        force_prepare_dump = true;
    }
    else{
      String force_prepare_dump_str = platform::getEnvironmentVariable("ARCANE_FORCE_PREPARE_DUMP");
      if (force_prepare_dump_str=="TRUE" || force_prepare_dump_str=="1" || force_prepare_dump_str=="true")
        force_prepare_dump = true;
    }
    if (force_prepare_dump){
      info() << "TimeLoopMng::doOneIteration(): Force prepareDump()";
      // TODO: vérifier si nécessaire et si oui le faire pour tous les sous-domaines.
      sd->defaultMesh()->prepareForDump();
    }
  }

  if (m_stop_time_loop){
    if (m_stop_has_error)
      return (-1);
    return (+1);
  }
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Effectue un repartitionnement des maillages.
 */
void TimeLoopMng::
_doMeshPartition()
{
  ISubDomain* sd = subDomain();

  // Détruit le gestionnaire de retour-arrière pour économiser de la mémoire.
  // Il sera reconstruit après le partitionnement.
  m_backward_mng->clear();

  Timer timer(sd,"TimeLoopMng::partitionMesh",Timer::TimerReal);
  Timer::Action ts_action(sd, "MeshesLoadBalance", true);

  for (IMeshPartitionerBase* mesh_partitioner : m_mesh_partitioner) {
    IMesh* mesh = mesh_partitioner->primaryMesh();
    Timer::Action ts_action2(sd, mesh->name(), true);
    {
      Timer::Sentry sentry(&timer);
      mesh->utilities()->partitionAndExchangeMeshWithReplication(mesh_partitioner, false);
    }
    info() << "Time spent to repartition the mesh (unit: second): "
           << timer.lastActivationTime();

    // Écrit dans les logs dans les infos sur la distribution du voisinage
    // TODO: pouvoir configurer cela et éventuellement ajouter des informations
    {
      IItemFamily* cell_family = mesh->cellFamily();
      IVariableSynchronizer* sync_info = cell_family->allItemsSynchronizer();
      auto communicating_ranks = sync_info->communicatingRanks();
      Int64 current_iteration = subDomain()->commonVariables().globalIteration();
      {
        JSONWriter json_writer(JSONWriter::FormatFlags::None);
        json_writer.beginObject();
        json_writer.write("Mesh", mesh->name());
        json_writer.write("Iteration", current_iteration);
        json_writer.write("CommunicatingRanks", communicating_ranks);
        json_writer.endObject();
        plog() << "MeshPartitionCommunicatingInfos:" << json_writer.getBuffer();
      }
    }
  }

  {
    Timer::Action ts_action1(sd, "OnMeshChangeEntryPoints", true);
    execOnMeshChangedEntryPoints();
  }

  m_mesh_partitioner.clear();

  // Affiche les statistiques d'exécution
  sd->timeStats()->dumpCurrentStats("MeshesLoadBalance");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_callSpecificEntryPoint()
{
  IEntryPoint* ep = m_entry_point_mng->findEntryPoint(m_specific_entry_point_name);
  info() << "Calling specific entry point: " << m_specific_entry_point_name;
  if (!ep)
    ARCANE_FATAL("No entry point named '{0}' found",m_specific_entry_point_name);
  ep->executeEntryPoint();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
registerTimeLoop(ITimeLoop * timeloop)
{
  ITraceMng* msg = traceMng();
  Trace::Setter mci(msg,m_message_class_name);

  const String& name = timeloop->name();

  log() << "Registering the time loop " << name;

  auto tl = m_time_loop_list.find(name);
  if (tl != m_time_loop_list.end())
    ARCANE_FATAL("The time loop '{0}' is defined twice",name);

  m_time_loop_list.insert(TimeLoopMap::value_type(name, timeloop));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
setUsedTimeLoop(const String& name)
{
  ITraceMng * msg = traceMng();
  Trace::Setter mci(msg,m_message_class_name);

  m_used_time_loop = nullptr;
  if (name.null())
    m_used_time_loop = m_default_time_loop;
  else{
    auto tl = m_time_loop_list.find(name);
    if (tl != m_time_loop_list.end())
      m_used_time_loop = tl->second;
  }

  if (!m_used_time_loop){
    info() << "Available time loops: ";
    for( const auto& tl : m_time_loop_list ){
      info() << "Time loop <" << tl.second->name() << ">";
    }
    ARCANE_FATAL("Unknown time loop '{0}'",name);
  }

  logdate() << "Using time loop " << name; 

  ISubDomain* sd = m_sub_domain;
  // Fusionne la configuration du sous-domaine avec celle de la boucle en temps.
  sd->configuration()->merge(m_used_time_loop->configuration());

  ScopedPtrT<IServiceLoader> service_loader(sd->mainFactory()->createServiceLoader());

  service_loader->loadModules(sd,false);

  _fillModuleFactoryMap();

  // Chargement des modules de la boucle en temps
  _fillModuleStateMap(m_used_time_loop);

  for( const auto& it : m_module_state_list ){
    const ModuleState & module_state = it.second;

    if (!module_state.m_is_optional || module_state.m_is_active){
      // creation du module
      if (!_createModule(module_state.m_alias))
        ARCANE_FATAL("The module \"{0}\" was not created.",module_state.m_alias);
    }
    else{
      info() << "The entry points of the module \"" 
             << module_state.m_alias << "\" won't be executed (inactive module).";
    }
  }

  _createSingletonServices(service_loader.get());

  m_used_time_loop_entry_points.clear();

  // Parcours des points d'entrée explicitement référencés dans la boucle en temps 
  // (m_used_time_loop) pour rechercher ceux qui correspondent aux points d'entrée
  // enregistrés dans les modules
  // Les points d'entrée non auto-load non référencés ne sont donc pas pris en charge.
  // Attention: dans la boucle en temps les points d'entrée sont référencés
  // par nom_module.nom_point_entrée alors que dans le fichier de données
  // ils sont référencés par alias_module.nom_point_entrée
  EntryPointList timeloop_entry_points;
  _processEntryPoints(timeloop_entry_points,
                      m_used_time_loop->entryPoints(ITimeLoop::WBuild),
                      ITimeLoop::WBuild);
  _processEntryPoints(timeloop_entry_points,
                      m_used_time_loop->entryPoints(ITimeLoop::WComputeLoop),
                      ITimeLoop::WComputeLoop);
  _processEntryPoints(timeloop_entry_points,
                      m_used_time_loop->entryPoints(ITimeLoop::WRestore),
                      ITimeLoop::WRestore);
  _processEntryPoints(timeloop_entry_points,
                      m_used_time_loop->entryPoints(ITimeLoop::WOnMeshChanged),
                      ITimeLoop::WOnMeshChanged);
  _processEntryPoints(timeloop_entry_points,
                      m_used_time_loop->entryPoints(ITimeLoop::WOnMeshRefinement),
                      ITimeLoop::WOnMeshRefinement);
  _processEntryPoints(timeloop_entry_points,
                      m_used_time_loop->entryPoints(ITimeLoop::WInit),
                      ITimeLoop::WInit);
  _processEntryPoints(timeloop_entry_points,
                      m_used_time_loop->entryPoints(ITimeLoop::WExit),
                      ITimeLoop::WExit);

  // Ajoute les autoload au début.
  EntryPointList entry_points(m_entry_point_mng->entryPoints());
  for( EntryPointCollection::Enumerator i(entry_points); ++i; ){
    IEntryPoint* ic = * i;
    if (ic->property() & IEntryPoint::PAutoLoadBegin)
      _addExecuteEntryPoint(ic);
  }

  // Ajoute les points d'entrées de la boucle de calcul.
  for (EntryPointCollection::Enumerator i(timeloop_entry_points); ++i; )
    _addExecuteEntryPoint(*i);

  { // Ajoute les autoload à la fin dans le sens inverse de leur déclaration
    EntryPointList auto_load_ends;
    for( EntryPointCollection::Enumerator i(entry_points); ++i; ){
      IEntryPoint* ic = *i;
      if (ic->property() & IEntryPoint::PAutoLoadEnd){
        auto_load_ends.add(ic);
      }
    }
    for( Integer i=0, s=auto_load_ends.count(); i<s; ++i){
      IEntryPoint * ic = auto_load_ends[s - 1 - i];
      _addExecuteEntryPoint(ic);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_createSingletonServices(IServiceLoader* service_loader)
{
  ISubDomain* sd = m_sub_domain;
  for( TimeLoopSingletonServiceInfoCollection::Enumerator i(m_used_time_loop->singletonServices()); ++i; ){
    const TimeLoopSingletonServiceInfo& ti = *i;
    String name = ti.name();
    bool is_found = service_loader->loadSingletonService(sd,name);
    if (!is_found){
      if (ti.isRequired())
        ARCANE_FATAL("Unable to find a singleton service named '{0}'",name);
      info() << "The optional singleton service named '" << name << "' was not found";
    }
    info() << "Loading singleton service '" << name << "'";
  }

  // Lecture des services spécifiés dans le jeu de données.
  ICaseMng* cm = m_sub_domain->caseMng();
  ICaseDocument* doc = cm->caseDocument();
  if (doc){
    XmlNode services_element = doc->servicesElement();
    String ustr_name("name");
    String ustr_active("active");
    XmlNodeList services = services_element.children("service");
    for( XmlNode x : services ) {
      String name = x.attrValue(ustr_name);
      XmlNode active_node = x.attr(ustr_active);
      bool is_active = true;
      if (!active_node.null())
        is_active = active_node.valueAsBoolean(true);
      if (is_active)
        service_loader->loadSingletonService(sd,name);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_processEntryPoints(EntryPointList& entry_points,
                    const TimeLoopEntryPointInfoCollection& entry_points_info,
                    const char* where)
{
  //TODO: Verifier que le nom du module spécifié dans la liste des modules à activer existe bien.
  for( TimeLoopEntryPointInfoCollection::Enumerator i(entry_points_info); ++i; ){
    const TimeLoopEntryPointInfo& entry_point_info = *i;
    const String& timeloop_call_name = entry_point_info.name();
    String entry_point_name;
    String module_name;
    _extractModuleAndEntryPointName(timeloop_call_name,module_name,entry_point_name);

    auto it = m_module_state_list.find(module_name);
    if (it == m_module_state_list.end())
      ARCANE_FATAL("No module named '{0}' is referenced",module_name);

    const ModuleState & module_state = it->second;
    if (!module_state.m_is_optional || module_state.m_is_active){
      if (!module_state.m_alias.null())
        module_name = module_state.m_alias;
      StringBuilder call_alias(module_name);
      call_alias += ".";
      call_alias += entry_point_name;

      IEntryPoint* entry_point = m_entry_point_mng->findEntryPoint(module_name,entry_point_name);
      log() << "Looking for entry point '" << call_alias << "'";
      if (!entry_point)
        ARCANE_FATAL("No entry point named '{0}' is referenced",call_alias);

      // il faut verifier que la propriete "where" du point d'entree est
      // compatible avec l'attribut "where" donne dans la boucle en temps
      String ep_where = entry_point->where();
      OStringStream msg;
      msg() << "The entry point '" << call_alias << "' declared \"" << ep_where
            << "\" can't be in the entry point list \""
            << where << "\" of the time loop";

      if (ep_where==IEntryPoint::WComputeLoop && where!=ITimeLoop::WComputeLoop)
        ARCANE_FATAL(msg.str());
      if (ep_where==IEntryPoint::WRestore && where!=ITimeLoop::WRestore)
        ARCANE_FATAL(msg.str());
      if (ep_where==IEntryPoint::WExit && where!=ITimeLoop::WExit)
        ARCANE_FATAL(msg.str());
      if ((ep_where==IEntryPoint::WInit || 
           ep_where==IEntryPoint::WContinueInit ||
           ep_where==IEntryPoint::WStartInit)
          && where != ITimeLoop::WInit)
        ARCANE_FATAL(msg.str());
      
      entry_points.add(entry_point);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_fillModuleStateMap(ITimeLoop* time_loop)
{
  // Analyse des infos de la boucle en temps
  //   1. liste des modules obligatoires
  for( StringCollection::Enumerator i(time_loop->requiredModulesName()); ++i; ){
    const String& module_name = *i;
    ModuleState ms(false, module_name);
    m_module_state_list.insert(ModuleStateMap::value_type(module_name, ms));
  }

  //   2. liste des modules optionnels
  for( StringCollection::Enumerator i(time_loop->optionalModulesName()); ++i; ){
    const String& module_name = *i;
    ModuleState ms(true, module_name);
    m_module_state_list.insert(ModuleStateMap::value_type(module_name,ms));
  }

  ICaseMng* cm = m_sub_domain->caseMng();
  // Remplissage de la liste d'état des modules à partir
  // des infos du fichier de données s'il existe
  // (Il n'y a pas de jeu de données par exemple lorsqu'on génére les infos
  // internes du code via l'option 'arcane_all_internal').
  ICaseDocument* doc = cm->caseDocument();
  if (!doc)
    return;

  XmlNode modules_node = doc->modulesElement();
  String ustr_module("module");
  String ustr_false("false");
  String ustr_name("name");
  String ustr_alias("alias");
  String ustr_active("active");
  for (XmlNode::const_iter i(modules_node); i(); ++i){
    if (i->name() != ustr_module)
      continue;
    XmlNode module_node = *i;
    String name = module_node.attrValue(ustr_name);
    String alias = module_node.attrValue(ustr_alias);

    // Regarde si le module est actif.
    XmlNode active_node = module_node.attr(ustr_active);
    bool active = true;
    if (!active_node.null())
      active = active_node.valueAsBoolean(true);

    // Complète les infos de la liste d'état

    // Regarde d'abord si 'name' correspond à un nom de module
    // dans la langue du JDD.
    auto ilang = m_lang_module_factory_map.find(name);
    if (ilang!=m_lang_module_factory_map.end())
      name = ilang->second->moduleName();

    auto it = m_module_state_list.find(name);
    if (it == m_module_state_list.end())
      // Lève une exception si le nom du module spécifié dans le JDD
      // ne correspond à aucun module enregistré.
      ARCANE_FATAL("Error in configuring active modules: no module named '{0}' is registered.",
                   name);

    ModuleState& ms = it->second;
    ms.m_is_active = active;
    if (!alias.null())
      ms.m_alias = alias;

    if (!ms.m_is_optional && !ms.m_is_active) {
      pwarning() << "The module \"" << ms.m_alias
                 << "\" can't be declared mandatory in the time loop"
                 << " while being inactive in the input data."
                 << " It's activity is therefore forced. ";
      ms.m_is_active = true;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_extractModuleAndEntryPointName(const String& timeloop_call_name,
                                String& module_name,
                                String& entry_point_name)
{
  std::string std_timeloop_call_name(timeloop_call_name.localstr());
  size_t index = std_timeloop_call_name.find_first_of('.');
  if (index==std::string::npos){
    ARCANE_FATAL("The string '{0}' is not a valid reference to an entry point (has to be of type "
                 "'module_name.entry_point_name)",timeloop_call_name);
  }
  std::string std_module_name = std_timeloop_call_name.substr(0, index);
  std::string std_entry_point_name = std_timeloop_call_name.substr(index+1);
  module_name = std_module_name;
  entry_point_name = std_entry_point_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit \a m_module_factory_map avec la liste des fabriques disponibles.
 */
void TimeLoopMng::
_fillModuleFactoryMap()
{
  String lang;
  ICaseDocumentFragment* doc = m_sub_domain->caseMng()->caseDocumentFragment();
  if (doc)
    lang = doc->language();
  ModuleFactoryInfoCollection module_factories(subDomain()->application()->moduleFactoryInfos());
  m_module_factory_map.clear();
  m_lang_module_factory_map.clear();
  for( ModuleFactoryInfoCollection::Enumerator i(module_factories); ++i; ){
    IModuleFactoryInfo* mfi = *i;

    const String& module_name = mfi->moduleName();
    if (m_module_factory_map.find(module_name)!=m_module_factory_map.end())
      ARCANE_FATAL("Two modules with same name '{0}'",module_name);
    m_module_factory_map.insert(std::make_pair(module_name,mfi));
    info(5) << "Registering module in factory map name=" << module_name;

    if (!lang.null()){
      const IServiceInfo* si = mfi->serviceInfo();
      String translated_name = si->tagName(lang);
      if (m_lang_module_factory_map.find(module_name)!=m_lang_module_factory_map.end()){
        // Envoie juste un avertissement car cela ne posera pas forcément problème.
        warning() << "Two modules with same translated name=" << translated_name
                  << " ignoring name=" << module_name;
      }
      else{
        m_lang_module_factory_map.insert(std::make_pair(translated_name,mfi));
        info(5) << "Registering module in lang factory map name=" << translated_name;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool TimeLoopMng::
_createModule(const String& module_name)
{
  auto x = m_module_factory_map.find(module_name);
  if (x!=m_module_factory_map.end()){
    IModuleFactoryInfo* mfi = x->second;
    Ref<IModule> module = mfi->createModule(m_sub_domain,m_sub_domain->defaultMeshHandle());
    if (module.get()){
      info() << "Loading module " << module->name()
             << " (Version " << module->versionInfo() << ")";
      return true;
    }
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Ajoute le point d'entrée \a s à la liste des points d'entrée.
 */
void TimeLoopMng::
_addExecuteEntryPoint(IEntryPoint* entry_point)
{
  log() << "Adding the entry point `" << entry_point->module()->name() << "::" << entry_point->name() << "' to the execution";

  String where = entry_point->where();
  if (where==IEntryPoint::WInit || where==IEntryPoint::WContinueInit || where==IEntryPoint::WStartInit){
    m_init_entry_points.add(entry_point);
  }
  else if (where==IEntryPoint::WBuild){
    m_build_entry_points.add(entry_point);
  }
  else if (where==IEntryPoint::WExit){
    m_exit_entry_points.add(entry_point);
  }
  else if (where == IEntryPoint::WRestore){
    m_restore_entry_points.add(entry_point);
  }
  else if (where == IEntryPoint::WOnMeshChanged){
    m_on_mesh_changed_entry_points.add(entry_point);
  }
  else if (where == IEntryPoint::WOnMeshRefinement){
    m_on_mesh_refinement_entry_points.add(entry_point);
  }
  else if (IEntryPoint::WComputeLoop){
    m_loop_entry_points.add(entry_point);
    m_current_entry_point = m_loop_entry_points.count();
  }
  m_used_time_loop_entry_points.add(entry_point);

  // Puisque le module a un point d'entrée utilisé, il est activé.
  IModule* c = entry_point->module();
  c->setUsed(true);
  {
    if (std::find(m_list_execute_module.begin(),m_list_execute_module.end(),c)==m_list_execute_module.end()){
      debug() << "Adding the module  `" << c->name() << "' to the execution";
      m_list_execute_module.add(c);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_dumpTimeInfos(JSONWriter& json_writer)
{
  IModuleMng * module_mng = subDomain()->moduleMng();

  EntryPointCollection entry_points = m_entry_point_mng->entryPoints();

  Real total_exec_time = cpuTimeUsed();

  {
    Real total_real_time = 0.0;
    Real compute_real_time = 0.0;
    json_writer.writeKey("EntryPoints");
    json_writer.beginArray();
    for( EntryPointCollection::Enumerator i(entry_points); ++i; ){
      IEntryPoint* ep = *i;
      Real s2 = ep->totalElapsedTime();
      {
        JSONWriter::Object jo(json_writer);
        json_writer.write("Name",ep->name());
        json_writer.write("TotalCpuTime",s2);
        json_writer.write("TotalElapsedTime",s2);
        json_writer.write("NbCall",(Int64)ep->nbCall());
        json_writer.write("Where",ep->where());
      }
      info(5) << "CPU_TIME where=" << ep->where() << " name=" << ep->name() << " S=" << s2;
      total_real_time += s2;
      if (ep->where()==IEntryPoint::WComputeLoop){
        compute_real_time += s2;
      }
    }
    json_writer.endArray();
    info(4) << "TOTAL_REAL_TIME COMPUTE=" << compute_real_time << " TOTAL=" << total_real_time;
  }


  const CommonVariables& scv = subDomain()->commonVariables();
  info() << "Information on the execution time";
  {
    Accelerator::IAcceleratorMng* acc_mng = m_sub_domain->acceleratorMng();
    if (acc_mng->isInitialized()){
      Accelerator::Runner* runner = acc_mng->defaultRunner();
      info() << " TotalRunner (" << runner->executionPolicy() << ") = "
             << runner->cumulativeCommandTime() << " seconds";
    }
  }
  info() << " TotalElapsed  = " << total_exec_time << " seconds";
  info() << " CumulativeElapsed = " << scv.globalElapsedTime()
         << " seconds (" << platform::timeToHourMinuteSecond(scv.globalElapsedTime()) << ")";
  info() << " T   = Total time spent in the function or in the module (s)";
  info() << " TC  = Total time spend per call (ms)";
  info() << " TCC = Total time spent per call and per cell (ns)";
  info() << " N   = Number of time the function was called";

  info() << " Use the clock time (elapsed) for the statistics";

  std::ostringstream o;
  std::ios_base::fmtflags f = o.flags(std::ios::right);
  Integer nb_cell = 0;
  IMesh * mesh = subDomain()->defaultMesh();
  if (mesh)
    nb_cell = mesh->nbCell();
  if (nb_cell == 0)
    nb_cell = 1;

  o << "\n              Name                            T        TC       TCC    %         N\n";

  for (ModuleCollection::Enumerator j(module_mng->modules()); ++j; ){
    IModule * module = * j;
    Real total_time_module = 0.;
    Real total_time_module_entry_point = 0.;
    for (EntryPointCollection::Enumerator i(entry_points); ++i; ){
      IEntryPoint * ic = * i;
      if (ic->module()!=module)
        continue;
      Integer nb_call = ic->nbCall();
      if (nb_call==0)
        continue;
      Real total_time = ic->totalElapsedTime();
      //if (math::isZero(total_time))
      //continue;
      const String& ep_name = ic->name();
      {
        Int64 l = ep_name.length();
        if (l > 36){
          o.write(ep_name.localstr(), 34);
          o << "...";
        }
        else{
          o.width(37);
          o << ep_name;
        }
      }
      o.width(10);
      Int64 z = Convert::toInt64(total_time);
      o << z;
      Real r = (1e3 * total_time) / nb_call;
      if (ic->where() == IEntryPoint::WComputeLoop){
        total_time_module += r;
        total_time_module_entry_point += total_time;
      }
      z = Convert::toInt64(r);
      o.width(10);
      o << z;
      r = (r * 1e6) / nb_cell;
      z = Convert::toInt64(r);
      o.width(10);
      o << z;
      //Probleme sur certaine machine : total_exec_time = 0
      if (total_exec_time>0)
        z = Convert::toInt64((100.0 * total_time) / total_exec_time);
      else
        z = Convert::toInt64((100.0 * total_time)) ;
      o.width(5);
      o << z;
      o.width(12);
      o << nb_call;
      o << '\n';
    }
    o << "--";
    o.width(33);
    o << module->name();
    o << "--";
    o.width(10);
    Int64 z = Convert::toInt64(total_time_module_entry_point * 1e-3);
    o << z;
    o.width(10);
    z = Convert::toInt64(total_time_module);
    o << z;
    Real r = (total_time_module * 1e6) / nb_cell;
    z = Convert::toInt64(r);
    o.width(10);
    o << z;
    o << '\n';
  }
  o.flags(f);

  info() << o.str();

  {
    IParallelMng* pm = m_sub_domain->parallelMng();
    if (pm->isParallel()){
      JSONWriter::Object jo(json_writer,"MessagePassingStats");
      Parallel::IStat* s = pm->stat();
      if (s){
        s->printCollective(pm);
        s->dumpJSON(json_writer);
      }
    }
  }
  {
    JSONWriter::Object jo(json_writer,"VariablesStats");
    m_sub_domain->variableMng()->dumpStatsJSON(json_writer);
  }
  {
    JSONWriter::Object jo(json_writer,"TimeStats");
    m_sub_domain->timeStats()->dumpStatsJSON(json_writer);
  }
  {
    IProfilingService* ps = platform::getProfilingService();
    if (ps){
      JSONWriter::Object jo(json_writer,"Profiling");
      ps->dumpJSON(json_writer);
    }
  }

  Item::dumpStats(traceMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real TimeLoopMng::
cpuTimeUsed() const
{
  EntryPointCollection entry_points = m_entry_point_mng->entryPoints();
  Real total_elapsed_time = 0.0;
  for( EntryPointCollection::Enumerator i(entry_points); ++i; ){
    Real s1 = (*i)->totalElapsedTime();
    total_elapsed_time += s1;
  }
  return total_elapsed_time;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool TimeLoopMng::
isDoingBackward()
{
  if (!m_backward_mng)
    _createOwnDefaultBackwardMng();
  return m_backward_mng->isLocked();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
goBackward()
{
  ITraceMng * msg = traceMng();
  Trace::Setter mci(msg,m_message_class_name);

  if (!m_backward_mng)
    _createOwnDefaultBackwardMng();

  m_backward_mng->goBackward();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
timeLoopsName(StringCollection& names) const
{
  for( ConstIterT<TimeLoopMap> i(m_time_loop_list); i(); ++i)
    names.add(i->first);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
timeLoops(TimeLoopCollection& time_loops) const
{
  for( ConstIterT<TimeLoopMap> i(m_time_loop_list); i(); ++i)
    time_loops.add(i->second);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITimeLoop * TimeLoopMng::
createTimeLoop(const String& name)
{
  IApplication * sm = m_sub_domain->application();
  return sm->mainFactory()->createTimeLoop(sm, name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
setStopReason(eTimeLoopStopReason reason)
{
  m_stop_reason = reason;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
stopComputeLoop(bool is_final_time,bool has_error)
{
  m_stop_time_loop = true;
  m_stop_has_error = has_error;
  if (is_final_time)
    m_final_time_reached = true;
  if (has_error)
    m_stop_reason = eTimeLoopStopReason::Error;
  else if (is_final_time)
    m_stop_reason = eTimeLoopStopReason::FinalTimeReached;
  // Si \a m_stop_reason n'est pas encore spécifié, indique qu'il n'y a pas
  // de raison spéciale
  if (m_stop_reason==eTimeLoopStopReason::NoStop)
    m_stop_reason = eTimeLoopStopReason::NoReason;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
registerActionMeshPartition(IMeshPartitionerBase* mesh_partitioner)
{
  if (!m_backward_mng)
    _createOwnDefaultBackwardMng();

  if (m_backward_mng->isLocked() || m_backward_mng->isBackwardEnabled()){
    info() << "Repartioning required but inactive due to a backward process active";
    return;
  }
  m_mesh_partitioner.add(mesh_partitioner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int TimeLoopMng::
doComputeLoop(Integer max_loop)
{
  m_nb_loop = 0;
  bool is_end = false;
  bool is_end_by_max_loop = false;
  int ret_val = 0;
  //m_alarm_timer_value = 0;
  //pwarning() << "Force le alarm_timer_value a zéro";
  if (m_alarm_timer_value>0){
    info() << "Set the timeout before alarm at " << m_alarm_timer_value << " seconds";
  }

  // Allocation d'un gestionnaire de retour-arrière par défaut
  if (!m_backward_mng)
    _createOwnDefaultBackwardMng();

  IProfilingService* ps = platform::getProfilingService();
  bool want_specific_profiling = false;
  // Regarde si on demande un profiling spécifique. Dans ce cas,
  // les modules et services gèrent eux même le profiling et donc
  // on ne démarre pas automatiquement le profiling au début de la
  // boucle de calcul
  if (!platform::getEnvironmentVariable("ARCANE_SPECIFIC_PROFILING").null()){
    info() << "Specific profiling activated";
    want_specific_profiling = true;
  }

  {
    // NOTE: arcaneGlobalMemoryInfo() peut changer au cours du calcul
    // donc il faut le récupérer à chaque fois qu'on en a besoin.
    IMemoryInfo* mem_info = arcaneGlobalMemoryInfo();
    String s = platform::getEnvironmentVariable("ARCANE_CHECK_MEMORY_BLOCK_SIZE_ITERATION");
    if (!s.null()){
      Int64 block_size = 0;
      bool is_bad = builtInGetValue(block_size,s);
      if (!is_bad && block_size>2){
        info() << "Set Memory StackTraceMinAllocSize to " << block_size;
        mem_info->setStackTraceMinAllocSize(block_size);
      }
    }
  }

  // Demarrage du profiling de message passing le cas echeant
  if (m_msg_pass_prof_srv.get())
    m_msg_pass_prof_srv->startProfiling();

  try{
    if (ps){
      if (!ps->isInitialized())
        ps->initialize();
      ps->reset();
    }
    Item::resetStats();
    // Désactive le profiling si demande spécifique
    if (want_specific_profiling)
      ps = nullptr;
    ProfilingSentryWithInitialize ps_sentry(ps);
    if (!m_specific_entry_point_name.null()){
      _callSpecificEntryPoint();
      is_end = true;
    }
    while (!is_end){
      if (max_loop!=0 && m_nb_loop>=max_loop){
        info()<<"===================================================";
        info()<<"====== MAXIMUM NUMBER OF ITERATION REACHED  =======";
        info()<<"===================================================";
        is_end_by_max_loop = true;
        m_stop_reason = eTimeLoopStopReason::MaxIterationReached;
        stopComputeLoop(false,false);
        break;
      }
      // Indique qu'on va s'arrêter par nombre max d'itération
      if (max_loop!=0 && (1+m_nb_loop)>=max_loop){
        m_stop_reason = eTimeLoopStopReason::MaxIterationReached;
      }
      ret_val = doOneIteration();
      IMemoryInfo* mem_info = arcaneGlobalMemoryInfo();
      if (mem_info && mem_info->isCollecting()){
        mem_info->endCollect();
        Integer iteration = subDomain()->commonVariables().globalIteration();
        mem_info->setIteration(iteration);

        std::ostringstream ostr;
        if (iteration>0)
          mem_info->printAllocatedMemory(ostr,iteration-1);
        if (iteration>1)
          mem_info->printAllocatedMemory(ostr,iteration-2);
        if (iteration>4)
          mem_info->printAllocatedMemory(ostr,iteration-5);
        info() << "ITERATION_MemoryInfo: " << ostr.str();
        mem_info->beginCollect();
      }
      subDomain()->variableMng()->synchronizerMng()->flushPendingStats();
      if (ret_val!=0)
        is_end = true;
      ++m_nb_loop;
    }
  }
  catch (const TimeoutException& ex){
    if (m_msg_pass_prof_srv.get())
      m_msg_pass_prof_srv->stopProfiling();
    IParallelMng * pm = subDomain()->parallelMng();
    pinfo() << "TIMEOUT " << Trace::Width(8) << pm->commRank() << "_RANK Infos:"<<ex.additionalInfo();
    traceMng()->flush();
    // Attend que tous les processeurs envoient le signal et affichent le
    // message précédent
    platform::sleep(40);
    throw;
  }
  if (m_verification_only_at_exit){
    info() << "Doing verification at exit";
    doVerification("AtExit");
  }
  {
    IProfilingService* ps2 = platform::getProfilingService();
    if (ps2)
      ps2->printInfos(true);
  }
  if (m_msg_pass_prof_srv.get())
    m_msg_pass_prof_srv->stopProfiling();
  if (IItemEnumeratorTracer::singleton())
    IItemEnumeratorTracer::singleton()->dumpStats();
  info() << "End of compute loop: reason=" << (int)m_stop_reason;
  if (is_end_by_max_loop)
    return 2;
  if (m_final_time_reached)
    return 1;
  if (ret_val<0){
    return ret_val;
  }
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopMng::
_resetTimer() const
{
  if (m_alarm_timer_value>0)
    platform::resetAlarmTimer(m_alarm_timer_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
