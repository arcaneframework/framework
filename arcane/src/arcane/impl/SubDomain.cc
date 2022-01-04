// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SubDomain.cc                                                (C) 2000-2021 */
/*                                                                           */
/* Gestionnaire du sous-domaine.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StdHeader.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/VersionInfo.h"
#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/MemoryInfo.h"
#include "arcane/utils/List.h"
#include "arcane/utils/Property.h"
#include "arcane/utils/TraceAccessor2.h"

#include "arcane/ISubDomain.h"
#include "arcane/IVariableMng.h"
#include "arcane/IModuleMng.h"
#include "arcane/IServiceMng.h"
#include "arcane/IModule.h"
#include "arcane/IModuleMaster.h"
#include "arcane/IEntryPointMng.h"
#include "arcane/IEntryPoint.h"
#include "arcane/IMeshReader.h"
#include "arcane/ITimerMng.h"
#include "arcane/IApplication.h"
#include "arcane/IGhostLayerMng.h"
#include "arcane/ArcaneException.h"
#include "arcane/CaseOptionsMain.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/Properties.h"
#include "arcane/IParallelMng.h"
#include "arcane/Directory.h"
#include "arcane/ITimeHistoryMng.h"
#include "arcane/ICaseMng.h"
#include "arcane/ICheckpointMng.h"
#include "arcane/IPropertyMng.h"
#include "arcane/ITimeStats.h"
#include "arcane/ITimeLoop.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IIOMng.h"
#include "arcane/IMainFactory.h"
#include "arcane/Timer.h"
#include "arcane/CommonVariables.h"
#include "arcane/XmlNode.h"
#include "arcane/ICaseDocument.h"
#include "arcane/IPhysicalUnitSystem.h"
#include "arcane/IPhysicalUnitSystemService.h"
#include "arcane/IService.h"
#include "arcane/IServiceInfo.h"
#include "arcane/ServiceUtils.h"
#include "arcane/IRessourceMng.h"
#include "arcane/ISession.h"
#include "arcane/IMeshStats.h"
#include "arcane/IParallelSuperMng.h"
#include "arcane/IParallelReplication.h"
#include "arcane/IServiceLoader.h"
#include "arcane/ICheckpointReader.h"
#include "arcane/IMeshPartitioner.h"
#include "arcane/IMeshWriter.h"
#include "arcane/ICaseMeshMasterService.h"
#include "arcane/ILoadBalanceMng.h"
#include "arcane/CaseNodeNames.h"
#include "arcane/ModuleBuildInfo.h"
#include "arcane/Observable.h"
#include "arcane/VariableCollection.h"
#include "arcane/SubDomainBuildInfo.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IMeshMng.h"
#include "arcane/MeshHandle.h"
#include "arcane/ObserverPool.h"
#include "arcane/ConfigurationPropertyReader.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/impl/ConfigurationReader.h"
#include "arcane/impl/internal/MeshMng.h"
#include "arcane/impl/internal/LegacyMeshBuilder.h"

#include "arcane/CaseOptionService.h"
#include "arcane/CaseOptionBuildInfo.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/AcceleratorRuntimeInitialisationInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshBuilderMaster
{
 public:
  MeshBuilderMaster(ICaseMng* cm,const String& default_name)
  : m_case_options(new Arcane::CaseOptions(cm,".")),
    m_mesh_service(CaseOptionBuildInfo(_configList(),"meshes", XmlNode(), default_name, 1, 1), false, false)
  {
    m_mesh_service.addAlternativeNodeName("fr","maillages");
  }
 private:
  Arcane::ICaseOptionList* _configList() const
  {
    return m_case_options->configList();
  }
 public:
  ICaseOptions* options() const { return m_case_options.get(); }
 public:
  void createMeshes()
  {
    m_mesh_service.instance()->createMeshes();
  }
 private:
  ReferenceCounter<ICaseOptions> m_case_options;
  CaseOptionServiceT<ICaseMeshMasterService> m_mesh_service;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation d'un gestionnaire de sous-domaine.
 *
 * Cette classe implémente l'interface ISubDomain.
 */
class SubDomain
: public ISubDomain
, public internal::TraceAccessor2
{
  ARCANE_DECLARE_PROPERTY_CLASS(SubDomain);

 public:

  //! Classe pour gérer la lecture/écriture des propriétés dans les protections/reprises
  class PropertyMngCheckpoint
  : public TraceAccessor
  {
   public:
    PropertyMngCheckpoint(ISubDomain* sd)
    : TraceAccessor(sd->traceMng()), m_sub_domain(sd),
      m_property_values(VariableBuildInfo(sd,"ArcaneProperties",IVariable::PPrivate))
    {
      init();
    }
   public:
    void init()
    {
      IVariableMng* vm = m_sub_domain->variableMng();
      m_observers.addObserver(this,
                              &PropertyMngCheckpoint::_notifyWrite,
                              vm->writeObservable());
      m_observers.addObserver(this,
                              &PropertyMngCheckpoint::_notifyRead,
                              m_property_values.variable()->readObservable());
    }
   private:
    void _notifyRead()
    {
      info(4) << "PropertyMngCheckpoint: READ";
      m_sub_domain->propertyMng()->readFrom(m_property_values);
    }
    void _notifyWrite()
    {
      info(4) << "PropertyMngCheckpoint: WRITE";
      m_sub_domain->propertyMng()->writeTo(m_property_values._internalTrueData()->_internalDeprecatedValue());
      m_property_values.variable()->syncReferences();
    }
   private:
    ISubDomain* m_sub_domain;
    ObserverPool m_observers;
    VariableArrayByte m_property_values;
  };

 public:
  
  SubDomain(ISession*,Ref<IParallelMng>,Ref<IParallelMng>,const String& filename,ByteConstArrayView bytes);

 protected:
  
  ~SubDomain() override;

 public:
	
  void build() override;
  void initialize() override;
  void destroy() override;
	
 public:

  IBase* objectParent() const override { return m_application; }
  String objectNamespaceURI() const override { return m_namespace_uri; }
  String objectLocalName() const override { return m_local_name; }
  VersionInfo objectVersion() const override { return VersionInfo(1,0,0); }

 public:

  IMainFactory* mainFactory() override { return m_application->mainFactory(); }
  ISession* session() const override { return m_session; }
  IServiceMng* serviceMng() const override { return m_service_mng.get(); }
  ITimeLoopMng* timeLoopMng() override { return m_time_loop_mng.get(); }
  IIOMng* ioMng() override { return m_io_mng.get(); }
  IVariableMng* variableMng() override { return m_variable_mng.get(); }
  IModuleMng* moduleMng() override { return m_module_mng.get(); }
  IEntryPointMng* entryPointMng() override { return m_entry_point_mng.get(); }
  ICaseMng* caseMng() override { return m_case_mng.get(); }
  ITimerMng* timerMng() const override { return m_timer_mng; }
  ICheckpointMng* checkpointMng() const override { return m_checkpoint_mng.get(); }
  IPropertyMng* propertyMng() const override { return m_property_mng.get(); }
  ITimeStats* timeStats() const override { return m_time_stats; }
  IRessourceMng* ressourceMng() const override { return m_application->ressourceMng(); }
  ITraceMng* traceMng() const override { return TraceAccessor::traceMng(); }
  IMemoryInfo* memoryInfo() const override { return m_memory_info.get(); }
  IPhysicalUnitSystem* physicalUnitSystem() override { return m_physical_unit_system.get(); }
  ILoadBalanceMng* loadBalanceMng() override { return m_lb_mng.get(); }
  IMeshMng* meshMng() const override { return m_mesh_mng.get(); }
  IModuleMaster* moduleMaster() const override { return m_module_master; }
  const IConfiguration* configuration() const override { return m_configuration.get(); }
  IConfiguration* configuration() override { return m_configuration.get(); }
  IAcceleratorMng* acceleratorMng() override { return m_accelerator_mng.get(); }

  Int32 subDomainId() const override { return m_parallel_mng->commRank(); }
  Int32 nbSubDomain() const override { return m_parallel_mng->commSize(); }
  void setIsContinue() override { m_is_continue = true; }
  bool isContinue() const override { return m_is_continue; }
  void dumpInfo(ostream&) override;
  void doInitModules() override;
  void doExitModules() override;
  IMesh* defaultMesh() override { return m_default_mesh_handle.mesh(); }
  const MeshHandle& defaultMeshHandle() override { return m_default_mesh_handle; }
  IMesh* mesh() override { return m_default_mesh_handle.mesh(); }
  IMesh* findMesh(const String& name,bool throw_exception) override;
  bool isInitialized() const override { return m_is_initialized; }
  void setIsInitialized() override;
  const ApplicationInfo& applicationInfo() const override { return m_application->applicationInfo(); }
  ICaseDocument* caseDocument() override { return m_case_mng->caseDocument(); }
  IApplication* application() override { return m_application; }
  void checkId(const String& where,const String& id) override;
  const String& caseFullFileName() const override { return m_case_full_file_name; }
  void setCaseFullFileName(const String& file_name) { m_case_full_file_name = file_name; }
  const String& caseName() const override { return m_case_name; }
  void fillCaseBytes(ByteArray& bytes) const override { bytes.copy(m_case_bytes); }
  void setCaseName(const String& case_name) override { m_case_name = case_name; }
  void setInitialPartitioner(IInitialPartitioner* partitioner) override
  {
    m_legacy_mesh_builder->m_initial_partitioner = partitioner;
  }
  void readCaseMeshes() override;
  void allocateMeshes() override;
  void readOrReloadMeshes() override;
  void initializeMeshVariablesFromCaseFile() override;
  void doInitMeshPartition() override;
  void addMesh(IMesh* mesh) override;
  ConstArrayView<IMesh*> meshes() const override;
  const CaseOptionsMain* caseOptionsMain() const override { return m_case_config; }
  IParallelMng* parallelMng() override { return m_parallel_mng.get(); }
  IParallelMng* allReplicaParallelMng() const override { return m_all_replica_parallel_mng.get(); }
  IThreadMng* threadMng() override { return m_parallel_mng->threadMng(); }
  const IDirectory& exportDirectory() const override { return m_export_directory; }
  void setExportDirectory(const IDirectory& dir) override { m_export_directory = dir; }
  const IDirectory& storageDirectory() const override { return m_storage_directory; }
  void setStorageDirectory(const IDirectory& dir) override { m_storage_directory = dir; }
  const IDirectory& listingDirectory() const override { return m_listing_directory; }
  void setListingDirectory(const IDirectory& dir) override { m_listing_directory = dir; }
  ITimeHistoryMng* timeHistoryMng() override { return m_time_history_mng.get(); }
  const CommonVariables& variablesCommon() const { return commonVariables(); }
  const CommonVariables& commonVariables() const override;
  void dumpInternalInfos(XmlNode& root) override;
  Integer meshDimension() const override;
  IObservable* onDestroyObservable() override { return &m_on_destroy_observable; }
  IDirectExecution* directExecution() const override { return m_direct_execution; }
  void setDirectExecution(IDirectExecution* v) override { m_direct_execution = v; }

 public:

  void _setLegacyMeshCreation(bool v){ m_is_create_default_mesh_v2 = !v; }
  bool isLegacyMeshCreation() const { return !m_is_create_default_mesh_v2; }

 public:

 protected:

  const char* _msgClassName() const { return "Init"; }

 private:

  ISession* m_session; //!< Session
  IApplication* m_application; //!< Gestionnaire principal
  Ref<IParallelMng> m_parallel_mng; //!< Gestionnaire du parallélisme
  Ref<IParallelMng> m_all_replica_parallel_mng; //!< Gestionnaire du parallélisme pour tous les réplicats
  ScopedPtrT<IIOMng> m_io_mng; //!< Gestionnaire des entrées/sorties
  ScopedPtrT<IMemoryInfo> m_memory_info; //!< Informations sur l'utilisation mémoire
  ScopedPtrT<IVariableMng> m_variable_mng; //!< Gestionnaire des variables
  ScopedPtrT<IModuleMng> m_module_mng; //!< Gestionnaire des modules
  ScopedPtrT<IEntryPointMng> m_entry_point_mng; //!< Gestionnaire des points d'entrée
  ScopedPtrT<ICaseMng> m_case_mng; //!< Gestionnaire du jeu de données
  ITimerMng* m_timer_mng; //!< Gestionnaire des timers
  ScopedPtrT<ICheckpointMng> m_checkpoint_mng; //!< Gestionnaire de protections
  Ref<IPropertyMng> m_property_mng; //!< Gestionnaire de propriétés
  ITimeStats* m_time_stats; //!< Statistiques sur les temps d'exécution
  ScopedPtrT<ITimeLoopMng> m_time_loop_mng; //!< Gestionnaire de la boucle en temps
  ScopedPtrT<IServiceMng> m_service_mng; //!< Gestionnaire des services
  ScopedPtrT<IPhysicalUnitSystem> m_physical_unit_system; //!< Système d'unité physique.
  String m_namespace_uri;
  String m_local_name;
  IModuleMaster* m_module_master; //!< Module maitre
  ScopedPtrT<ITimeHistoryMng> m_time_history_mng; //!< Gestionnaire d'historique
  ScopedPtrT<MeshMng> m_mesh_mng;
  MeshHandle m_default_mesh_handle;
  bool m_is_initialized; //!< \a true si initialisé
  String m_case_full_file_name; //!< Chemin d'accès du cas
  String m_case_name; //!< Nom du cas
  ByteUniqueArray m_case_bytes; //!< Données du cas.
  CaseOptionsMain* m_case_config; //!< Config du cas
  Directory m_export_directory; //!< Répertoire d'exportation
  Directory m_storage_directory; //!< Répertoire d'archivage
  Directory m_listing_directory; //!< Répertoire des listings
  Observable m_on_destroy_observable; //!< Observable lors d'une destruction
  bool m_is_continue;
  IDirectExecution* m_direct_execution;
  ScopedPtrT<ILoadBalanceMng> m_lb_mng; //!< Gestionnaire de caracteristiques pour l'equilibrage
  ScopedPtrT<IConfiguration> m_configuration; //!< Configuration
  bool m_is_create_default_mesh_v2;
  ScopedPtrT<PropertyMngCheckpoint> m_property_mng_checkpoint;
  ScopedPtrT<LegacyMeshBuilder> m_legacy_mesh_builder;
  //! Indique si on utilise le mécanisme de service pour lire le maillage
  bool m_has_mesh_service = false;
  Ref<ICaseMeshMasterService> m_case_mesh_master_service;
  ObserverPool m_observers;
  Ref<IAcceleratorMng> m_accelerator_mng;

 private:

  void _doInitialPartition();
  void _doInitialPartitionForMesh(IMesh* mesh,const String& service_name,bool is_required);
  void _notifyWriteCheckpoint();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static ISubDomain* global_sub_domain = 0;

/* HP: ARCANE_IMPL_EXPORT en attendant de trouver une solution 
 * pour accéder aux traces hors d'un service ou module */
extern "C" ARCANE_IMPL_EXPORT ISubDomain* _arcaneGetDefaultSubDomain()
{
  return global_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ISubDomain*
arcaneCreateSubDomain(ISession* session,const SubDomainBuildInfo& sdbi)
{
  Ref<IParallelMng> pm = sdbi.parallelMng();
  Ref<IParallelMng> all_replica_pm = sdbi.allReplicaParallelMng();
  String case_file_name = sdbi.caseFileName();
  ByteConstArrayView bytes = sdbi.caseBytes();

  ITraceMng* tm = pm->traceMng();
  StringBuilder trace_id;
  trace_id += String::fromNumber(pm->commRank());
  if (all_replica_pm!=pm){
    trace_id += ",r";
    trace_id += pm->replication()->replicationRank();
  }
  trace_id += ",";
  trace_id += platform::getHostName();
  tm->setTraceId(trace_id.toString());

  SubDomain* sd = new SubDomain(session,pm,all_replica_pm,case_file_name,bytes);
  sd->build();
  //GG: l'init se fait par l'appelant
  //mng->initialize();
  global_sub_domain = sd;
  return sd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SubDomain::
SubDomain(ISession* session,Ref<IParallelMng> pm,Ref<IParallelMng> all_replica_pm,
          const String& case_file_name,ByteConstArrayView bytes)
: TraceAccessor2(pm->traceMng())
, m_session(session)
, m_application(session->application())
, m_parallel_mng(pm)
, m_all_replica_parallel_mng(all_replica_pm)
, m_timer_mng(nullptr)
, m_time_stats(nullptr)
, m_namespace_uri(arcaneNamespaceURI())
, m_module_master(nullptr)
, m_time_history_mng(nullptr)
, m_is_initialized(false)
, m_case_full_file_name(case_file_name)
, m_case_name("unknown")
, m_case_bytes(bytes)
, m_case_config(nullptr)
, m_storage_directory(".")
, m_is_continue(false)
, m_direct_execution(nullptr)
, m_is_create_default_mesh_v2(false)
{
  m_local_name = "SubDomain";
  m_memory_info = new MemoryInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Cet appel doit être fait depuis une section critique.
void SubDomain::
build()
{
  IMainFactory* mf = m_application->mainFactory();
  m_physical_unit_system = m_application->getPhysicalUnitSystemService()->createStandardUnitSystem();
  m_configuration = m_application->configurationMng()->defaultConfiguration()->clone();

  m_accelerator_mng = mf->createAcceleratorMngRef(traceMng());
  m_property_mng = mf->createPropertyMngReference(this);
  m_io_mng = mf->createIOMng(parallelMng());
  m_variable_mng = mf->createVariableMng(this);
  m_mesh_mng = new MeshMng(m_application,m_variable_mng.get());
  m_default_mesh_handle = m_mesh_mng->createMeshHandle("Mesh0");
  m_service_mng = mf->createServiceMng(this);
  m_checkpoint_mng = mf->createCheckpointMng(this);
  m_module_mng = mf->createModuleMng(this);
  m_entry_point_mng = mf->createEntryPointMng(this);
  m_case_mng = mf->createCaseMng(this);
  m_timer_mng = m_parallel_mng->timerMng();
  m_time_stats = m_parallel_mng->timeStats();
  m_time_loop_mng = mf->createTimeLoopMng(this);
  m_legacy_mesh_builder = new LegacyMeshBuilder(this,m_default_mesh_handle);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
initialize()
{
  // Initialisation du module parallèle
  // TODO: a supprimer car plus utile
  m_parallel_mng->initialize();

  {
    // Initialise le runner par défaut en fonction des paramètres donnés par
    // l'utilisateur.
    IApplication* app = application();
    m_accelerator_mng->initialize(app->acceleratorRuntimeInitialisationInfo());
  }

  IMainFactory* mf = m_application->mainFactory();

  m_time_loop_mng->build();

  m_has_mesh_service = false;

  // Lit le jeu de données.
  // Il s'agit uniquement de s'assurer de la validité syntaxique
  // du jeu de données. Les options ne sont pas lues.
  // Cela doit être fait avant la création du maillage
  // car l'instance du maillage dépend du jeu de données
  ICaseDocument* case_document = nullptr;
  if (!m_case_bytes.empty()){
    case_document = m_case_mng->readCaseDocument(m_case_full_file_name,m_case_bytes);
    if (!case_document)
      ARCANE_FATAL("Can not read case options");
    // Ajoute à la configuration celle présente dans le jeu de données.
    {
      ConfigurationReader cr(traceMng(),m_configuration.get());
      cr.addValuesFromXmlNode(case_document->configurationElement(),ConfigurationReader::P_CaseDocument);
    }
  }

  properties::readFromConfiguration(m_configuration.get(),*this);

  if (case_document){
    // Regarde s'il y a un tag <meshes> indiquant que la création et
    // la lecture du maillage sera gérée par un service
    XmlNode meshes_elem = case_document->meshesElement();
    if (!meshes_elem.null()){
      info() << "Using mesh service to create and allocate meshes";
      m_has_mesh_service = true;
    }
    // TODO: dans ce cas vérifier qu'on n'a pas l'élément 'maillage' historique.
  }
  // Créé le maillage par défaut. Il faut le faire avant la création
  // des services car ces derniers peuvent en avoir besoin.
  if (!m_has_mesh_service)
    if (!m_is_create_default_mesh_v2)
      m_legacy_mesh_builder->createDefaultMesh();

  ScopedPtrT<IServiceLoader> service_loader(mf->createServiceLoader());
  service_loader->loadSubDomainServices(this);

  // Le module maître doit toujours être créé avant les autres pour que
  // ses points d'entrées soient appelés en premier et dernier
  m_module_master = mf->createModuleMaster(this);
  m_time_history_mng  = mf->createTimeHistoryMng(this);

  service_loader->initializeModuleFactories(this);
  m_lb_mng = mf->createLoadBalanceMng(this);

  m_property_mng_checkpoint = new PropertyMngCheckpoint(this);

  m_observers.addObserver(this,
                          &SubDomain::_notifyWriteCheckpoint,
                          m_checkpoint_mng->writeObservable());
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SubDomain::
~SubDomain()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
destroy()
{
  m_case_mesh_master_service.reset();

  m_property_mng_checkpoint = nullptr;

  m_on_destroy_observable.notifyAllObservers();
  m_on_destroy_observable.detachAllObservers();

  if (m_application->hasGarbageCollector())
    return;

  m_module_master = 0;
  
  m_module_mng->removeAllModules();
  m_service_mng = nullptr;
  m_time_history_mng = nullptr;
  m_mesh_mng->destroyMeshes();
  m_time_loop_mng = nullptr;
  m_case_mng = nullptr;
  m_entry_point_mng = nullptr;
  m_module_mng = nullptr;
  m_io_mng = nullptr;

  m_lb_mng = nullptr;
  m_property_mng.reset();

  // Comme tous les objets peuvent contenir des variables, il faut être
  // certain de détruire le gestionnaire de variables en dernier.
  m_variable_mng->removeAllVariables();
  m_variable_mng = nullptr;

  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
dumpInfo(ostream& o)
{
  m_module_mng->dumpList(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocation des maillages.
 *
 * Cette méthode gère uniquement la construction des instance IMesh.
 * La lecture de ces derniers se fait dans readOrReloadMeshes().
 */
void SubDomain::
allocateMeshes()
{
  Timer::Action ts_action2(this,"AllocateMesh");
  Trace::Setter mci(traceMng(),_msgClassName());

  // Vérifie que personne n'a fait d'appel à addMesh() avant cette méthode
  // car les maillages seront déréférencés.
  // NOTE: on garde cela pour des raisons de compatibilité mais il serait
  // préférable d'empêcher les appels à addMesh() avant d'avoir alloué
  // les maillages.
  // NOTE (2): depuis l'utilisation des MeshHandle (juin 2019), les maillages
  // déjà créés ne sont plus supprimés lors de l'appel à cette fonction
  // donc il n'y a pas besoin de faire un fatal si des maillages existent déjà.
  // On laisse le code en cas de problème mais si tout est OK on le supprimera
  // prochainement.
#if 0
  for( IMesh* mesh : m_mesh_mng->meshes() ){
    if (mesh!=m_default_mesh_handle.mesh())
      ARCANE_FATAL("Mesh named '{0}' created before call to SubDomain::allocateMeshes()",
                   mesh->name());
  }
#endif

  if (m_has_mesh_service){
    info() << "** Reading mesh from mesh service";
    CaseNodeNames* cnn = caseMng()->caseDocument()->caseNodeNames();
    String default_service_name = "ArcaneCaseMeshMasterService";

    // NOTE: cet objet sera détruit par caseMng()
    ICaseOptions* opt = new CaseOptions(caseMng(),cnn->meshes);
    ServiceBuilder<ICaseMeshMasterService> sb(application(),opt);
    Ref<ICaseMeshMasterService> mbm = sb.createReference(default_service_name);
    m_case_mesh_master_service = mbm;
    m_case_mng->_internalReadOneOption(mbm->_options(),true);
    mbm->createMeshes();
  }
  else{
    if (m_is_create_default_mesh_v2)
      m_legacy_mesh_builder->createDefaultMesh();

    m_legacy_mesh_builder->allocateMeshes();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
readOrReloadMeshes()
{
  logdate() << "Initialisation du code.";

  Integer nb_mesh = m_mesh_mng->meshes().size();
  info() << " nb_mesh_created=" << nb_mesh
         << " is_continue?=" << m_is_continue;

  A_INFO("Test: {1}",A_TR2("nb_mesh_created",nb_mesh),A_TR(m_is_continue));

  //info() << format4("Test: {1}",{A_PR2("nb_mesh_created",nb_mesh),A_PR(m_is_continue)});
  //info() << format5(A_PR2("nb_mesh_created",nb_mesh),A_PR(m_is_continue));

  if (m_is_continue){
    for( Integer z=0; z<nb_mesh; ++z ){
      IPrimaryMesh* mesh = m_mesh_mng->getPrimaryMesh(z);
      mesh->reloadMesh();
    }
  }
  else{
    if (m_has_mesh_service){
      m_case_mesh_master_service->allocateMeshes();
    }
    else
      m_legacy_mesh_builder->readMeshes();
  }

  //warning() << "CODE CHANGED IN DEV-X10. NEED TEST";
  //mesh->computeTiedInterfaces(mbi.m_xml_node);
  //! AMR : done in factory. This method is removed from IMesh.
  //      mesh->readAmrActivator(mbi.m_xml_node);

  // Vérifie la cohérence du maillage
  for( Integer z=0; z<nb_mesh; ++z ){
    IMesh* mesh = m_mesh_mng->getMesh(z);
    mesh->checkValidMesh();
    mesh->nodesCoordinates().setUpToDate();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
addMesh(IMesh* mesh)
{
  m_mesh_mng->addMesh(mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<IMesh*> SubDomain::
meshes() const
{
  return m_mesh_mng->meshes();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
initializeMeshVariablesFromCaseFile()
{
  if (!m_has_mesh_service)
    m_legacy_mesh_builder->initializeMeshVariablesFromCaseFile();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
doInitMeshPartition()
{
  if (!m_is_continue){
    IInitialPartitioner* init_part = m_legacy_mesh_builder->m_initial_partitioner.get();
    if (init_part){
      info() << "Using custom initial partitioner";
      init_part->partitionAndDistributeMeshes(m_mesh_mng->meshes());
    }
    else{
      if (m_case_mesh_master_service.get()){
        m_case_mesh_master_service->partitionMeshes();
      }
      else
        if (m_legacy_mesh_builder->m_use_internal_mesh_partitioner)
          _doInitialPartition();
    }
  }

  // Affichage des informations du maillage
  for( IMesh* mesh : m_mesh_mng->meshes() ){
    ScopedPtrT<IMeshStats> mh(IMeshStats::create(traceMng(),mesh,parallelMng()));
    mh->dumpStats();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Appelle les points d'entrée d'initialisation du module.
 */
void SubDomain::
doInitModules()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
_doInitialPartition()
{
  // NOTE: Ce code n'est utilisé que par le mécanisme historique de création
  // de maillage. Lorsqu'on utilise les maillages pour les services, le code
  // correspondant est dans ArcaneCaseMeshService.

  String test_service = "MeshPartitionerTester";

  for( IMesh* mesh : m_mesh_mng->meshes() ){
    bool is_mesh_allocated = mesh->isAllocated();
    info() << "InitialPartitioning mesh=" << mesh->name() << " is_allocated?=" << is_mesh_allocated;
    if (!is_mesh_allocated)
      continue;
    // Comme 'parmetis' n'aime pas repartitionner un maillage si un des sous-domaines
    // est vide (ce qui est le cas si un seul processeur créé les mailles), il
    // faut d'abord utiliser le partitionneur basique qui répartit les mailles
    // entre les sous-domaines, puis 'parmetis' s'il est présent
    if (m_legacy_mesh_builder->m_use_partitioner_tester) {
      Int64 nb_cell = mesh->nbCell();
      Int64 min_nb_cell = parallelMng()->reduce(Parallel::ReduceMin,nb_cell);
      info() << "Min nb cell=" << min_nb_cell;
      if (min_nb_cell==0)
        _doInitialPartitionForMesh(mesh,test_service,true);
      else
        info() << "Mesh name=" << mesh->name() << " have cells. Do not use " << test_service;
    }
    else
      info() << "No basic partition first needed";
    _doInitialPartitionForMesh(mesh,m_legacy_mesh_builder->m_internal_partitioner_name,false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
_doInitialPartitionForMesh(IMesh* mesh,const String& service_name,bool is_required)
{
  info() << "DoInitialPartition. mesh=" << mesh->name() << " service=" << service_name;

  String lib_name = service_name;

  IMeshPartitionerBase* mesh_partitioner_base = nullptr;
  Ref<IMeshPartitionerBase> mesh_partitioner_base_ref;
  Ref<IMeshPartitioner> mesh_partitioner_ref;

  ServiceBuilder<IMeshPartitionerBase> sbuilder(this);
  mesh_partitioner_base_ref = sbuilder.createReference(service_name,mesh,SB_AllowNull);
  mesh_partitioner_base = mesh_partitioner_base_ref.get();

  if (!mesh_partitioner_base){
    // Si pas trouvé, recherche avec l'ancienne interface 'IMeshPartitioner' pour des
    // raisons de compatibilité
    pwarning() << "No implementation for 'IMeshPartitionerBase' interface found. "
               << "Searching implementation for legacy 'IMeshPartitioner' interface";
    ServiceBuilder<IMeshPartitioner> sbuilder_legacy(this);
    mesh_partitioner_ref = sbuilder_legacy.createReference(service_name,mesh,SB_AllowNull);
    if (mesh_partitioner_ref.get())
      mesh_partitioner_base = mesh_partitioner_ref.get();
  }

  if (!mesh_partitioner_base){
    // Si pas trouvé, récupère la liste des valeurs possibles et les affiche.
    StringUniqueArray valid_names;
    sbuilder.getServicesNames(valid_names);
    String valid_values = String::join(", ",valid_names);
    String msg = String::format("The specified service for the initial mesh partitionment ({0}) "
                                "is not available (valid_values={1}). This service has to implement "
                                "interface Arcane::IMeshPartitionerBase",
                                lib_name,valid_values);
    if (is_required){
      pfatal() << msg;
    }
    else{
      pwarning() << msg;
    }
		return;
  }

  bool is_dynamic = mesh->isDynamic();
  mesh->modifier()->setDynamic(true);
  mesh->utilities()->partitionAndExchangeMeshWithReplication(mesh_partitioner_base,true);
  mesh->modifier()->setDynamic(is_dynamic);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
doExitModules()
{
  if (!m_is_initialized)
    ARCANE_FATAL("Uninitialized session");

  logdate() << "Execution of the end of compute entry points";

  // Sauve les données du gestionnaire d'historique si demandé
  if (caseOptionsMain()->doTimeHistory())
    m_time_history_mng->dumpHistory(true);

  m_time_loop_mng->execExitEntryPoints();

  m_parallel_mng->printStats();

  traceMng()->flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
checkId(const String& where,const String& id)
{
  Int64 len = id.length();
  const char* str = id.localstr();
  if (len==0 || !str)
    throw BadIDException(where,id);

  if (!isalpha(str[0]))
    throw BadIDException(where,id);
  for( Int64 i=1; i<len; ++i )
    if (!isalpha(str[i]) && !isdigit(str[i]) && str[i]!='_' && str[i]!='.')
      throw BadIDException(where,id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
readCaseMeshes()
{
  Trace::Setter mci(traceMng(),_msgClassName());

  info() << "Reading the case `" << m_case_full_file_name << "'";

  ICaseDocument* case_doc = caseDocument();
  if (!case_doc)
    ARCANE_FATAL("No input data");

  m_legacy_mesh_builder->readCaseMeshes();
  m_case_config = m_module_master->caseoptions();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const CommonVariables& SubDomain::
commonVariables() const
{
  if (!m_module_master)
    throw BadReferenceException("module master in SubDomain::variablesCommon()");
  return *m_module_master->commonVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
dumpInternalInfos(XmlNode& root)
{
  // Info sur la boucle en temps utilisée.
  {
    ITimeLoopMng* tlm = timeLoopMng();
    ITimeLoop* time_loop = tlm->usedTimeLoop();
    XmlElement time_loop_elem(root,"timeloopinfo");
    XmlElement title(time_loop_elem,"title",time_loop->title());
    XmlElement description(time_loop_elem,"description",time_loop->description());

    String ustr_userclass("userclass");
    for( StringCollection::Enumerator j(time_loop->userClasses()); ++j; )
      XmlElement elem(time_loop_elem,ustr_userclass,*j);
  }

  VariableRefList var_ref_list;
  XmlElement modules(root,"modules");
  
  String ustr_module("module");
  String ustr_name("name");
  String ustr_activated("activated");
  String ustr_true("true");
  String ustr_false("false");
  String ustr_variable("variable");
  String ustr_variable_ref("variable-ref");
  String ustr_ref("ref");
  String ustr_datatype("datatype");
  String ustr_dimension("dimension");
  String ustr_kind("kind");
  String ustr_caseblock("caseblock");
  String ustr_tagname("tagname");

  // Liste des modules avec les variables qu'ils utilisent.
  for( ModuleCollection::Enumerator i(moduleMng()->modules()); ++i; ){
    XmlElement module_element(modules,ustr_module);
    module_element.setAttrValue(ustr_name,(*i)->name());
    bool is_activated = (*i)->used();
    module_element.setAttrValue(ustr_activated,is_activated ? ustr_true : ustr_false);
    var_ref_list.clear();
    variableMng()->variables(var_ref_list,*i);
    for( VariableRefList::Enumerator j(var_ref_list); ++j; ){
      XmlElement variable_element(module_element,ustr_variable_ref);
      variable_element.setAttrValue(ustr_ref,(*j)->name());
    }
  }

  // Liste des variables.
  XmlElement variables(root,"variables");
  VariableCollection var_prv_list = variableMng()->variables();
  for( VariableCollection::Enumerator j(var_prv_list); ++j; ){
    XmlElement elem(variables,ustr_variable);
    IVariable* var = *j;
    String dim = String::fromNumber(var->dimension());
    elem.setAttrValue(ustr_name,var->name());
    elem.setAttrValue(ustr_datatype,dataTypeName(var->dataType()));
    elem.setAttrValue(ustr_dimension,dim);
    elem.setAttrValue(ustr_kind,itemKindName(var->itemKind()));
  }

  // Liste des blocs d'options
  ICaseMng* cm = caseMng();
  CaseOptionsCollection blocks = cm->blocks();
  XmlElement blocks_elem(root,"caseblocks");
  for( CaseOptionsCollection::Enumerator i(blocks); ++i; ){
    ICaseOptions* block = *i;
    XmlElement block_elem(blocks_elem,ustr_caseblock);
    block_elem.setAttrValue(ustr_tagname,block->rootTagName());
    IModule* block_module = block->caseModule();
    if (block_module)
      block_elem.setAttrValue(ustr_module,block_module->name());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer SubDomain::
meshDimension() const
{
  return m_default_mesh_handle.mesh()->dimension();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* SubDomain::
findMesh(const String& name,bool throw_exception)
{
  return m_mesh_mng->findMesh(name,throw_exception);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
_notifyWriteCheckpoint()
{
  info(4) << "SubDomain::_notifyWriteCheckpoint()";
  Properties time_stats_properties(propertyMng(),"TimeStats");
  timeStats()->saveTimeValues(&time_stats_properties);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
setIsInitialized()
{
  m_is_initialized = true;
  info(4) << "SubDomain::setIsInitialized()";
  Properties time_stats_properties(propertyMng(),"TimeStats");
  timeStats()->mergeTimeValues(&time_stats_properties);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename V> void SubDomain::
_applyPropertyVisitor(V& p)
{
  auto b = p.builder();

  p << b.addBool("LegacyMeshCreation")
        .addDescription("Using legacy mesh creation")
        .addGetter([](auto a) { return a.x.isLegacyMeshCreation(); })
        .addSetter([](auto a) { a.x._setLegacyMeshCreation(a.v); });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_PROPERTY_CLASS(SubDomain,());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
