// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SubDomain.cc                                                (C) 2000-2025 */
/*                                                                           */
/* Subdomain manager.                                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StdHeader.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/VersionInfo.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/MemoryInfo.h"
#include "arcane/utils/List.h"
#include "arcane/utils/TraceAccessor2.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/IProcessorAffinityService.h"
#include "arcane/utils/IProfilingService.h"
#include "arccore/common/internal/Property.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IModuleMng.h"
#include "arcane/core/IServiceMng.h"
#include "arcane/core/IModule.h"
#include "arcane/core/IModuleMaster.h"
#include "arcane/core/IEntryPointMng.h"
#include "arcane/core/IEntryPoint.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/ITimerMng.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/ArcaneException.h"
#include "arcane/core/CaseOptionsMain.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/Properties.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/Directory.h"
#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/ICheckpointMng.h"
#include "arcane/core/IPropertyMng.h"
#include "arcane/core/ITimeStats.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/Timer.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/IPhysicalUnitSystem.h"
#include "arcane/core/IPhysicalUnitSystemService.h"
#include "arcane/core/ISession.h"
#include "arcane/core/IMeshStats.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/IServiceLoader.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/ICaseMeshMasterService.h"
#include "arcane/core/ILoadBalanceMng.h"
#include "arcane/core/CaseNodeNames.h"
#include "arcane/core/Observable.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/SubDomainBuildInfo.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/ObserverPool.h"
#include "arcane/core/parallel/IStat.h"

#include "arcane/core/internal/ConfigurationPropertyReader.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/internal/ICaseMngInternal.h"
#include "arcane/core/internal/IParallelMngInternal.h"
#include "arcane/core/internal/IVariableMngInternal.h"

#include "arcane/impl/ConfigurationReader.h"
#include "arcane/impl/internal/MeshMng.h"
#include "arcane/impl/internal/LegacyMeshBuilder.h"

#include "arcane/core/CaseOptionService.h"
#include "arcane/core/CaseOptionBuildInfo.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/DeviceInfo.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/AcceleratorRuntimeInitialisationInfo.h"
#include "arcane/accelerator/core/IDeviceInfoList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshBuilderMaster
{
 public:

  MeshBuilderMaster(ICaseMng* cm, const String& default_name)
  : m_case_options(new Arcane::CaseOptions(cm, "."))
  , m_mesh_service(CaseOptionBuildInfo(_configList(), "meshes", XmlNode(), default_name, 1, 1), false, false)
  {
    m_mesh_service.addAlternativeNodeName("fr", "maillages");
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
 * \brief Implementation of a subdomain manager.
 *
 * This class implements the ISubDomain interface.
 */
class SubDomain
: public ISubDomain
, public internal::TraceAccessor2
{
  ARCANE_DECLARE_PROPERTY_CLASS(SubDomain);

 public:

  //! Class to manage the reading/writing of properties in checkpoints/rollbacks
  class PropertyMngCheckpoint
  : public TraceAccessor
  {
   public:

    explicit PropertyMngCheckpoint(ISubDomain* sd)
    : TraceAccessor(sd->traceMng())
    , m_sub_domain(sd)
    , m_property_values(VariableBuildInfo(sd, "ArcaneProperties", IVariable::PPrivate))
    {
      init();
    }

   public:

    void init()
    {
      m_observers.addObserver(this,
                              &PropertyMngCheckpoint::_notifyWrite,
                              m_property_values.variable()->writeObservable());
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

  SubDomain(ISession*, Ref<IParallelMng>, Ref<IParallelMng>, const String& filename, ByteConstArrayView bytes);

 public:

  void build() override;
  void initialize() override;
  void destroy() override;

 public:

  IBase* objectParent() const override { return m_application; }
  String objectNamespaceURI() const override { return m_namespace_uri; }
  String objectLocalName() const override { return m_local_name; }
  VersionInfo objectVersion() const override { return VersionInfo(1, 0, 0); }

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
  void dumpInfo(std::ostream&) override;
  void doInitModules() override;
  void doExitModules() override;
  IMesh* defaultMesh() override { return m_default_mesh_handle.mesh(); }
  const MeshHandle& defaultMeshHandle() override { return m_default_mesh_handle; }
  IMesh* mesh() override { return m_default_mesh_handle.mesh(); }
  IMesh* findMesh(const String& name, bool throw_exception) override;
  bool isInitialized() const override { return m_is_initialized; }
  void setIsInitialized() override;
  const ApplicationInfo& applicationInfo() const override { return m_application->applicationInfo(); }
  ICaseDocument* caseDocument() override { return m_case_mng->caseDocument(); }
  IApplication* application() override { return m_application; }
  void checkId(const String& where, const String& id) override;
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

  void _setLegacyMeshCreation(bool v) { m_is_create_default_mesh_v2 = !v; }
  bool isLegacyMeshCreation() const { return !m_is_create_default_mesh_v2; }

 public:
 protected:

  const char* _msgClassName() const { return "Init"; }

 private:

  ISession* m_session; //!< Session
  IApplication* m_application; //!< Main manager
  Ref<IParallelMng> m_parallel_mng; //!< Parallelism manager
  Ref<IParallelMng> m_all_replica_parallel_mng; //!< Parallelism manager for all replicas
  ScopedPtrT<IIOMng> m_io_mng; //!< Input/output manager
  ScopedPtrT<IMemoryInfo> m_memory_info; //!< Memory usage information
  ScopedPtrT<IVariableMng> m_variable_mng; //!< Variable manager
  ScopedPtrT<IModuleMng> m_module_mng; //!< Module manager
  ScopedPtrT<IEntryPointMng> m_entry_point_mng; //!< Entry point manager
  Ref<ICaseMng> m_case_mng; //!< Case data manager
  ITimerMng* m_timer_mng; //!< Timer manager
  ScopedPtrT<ICheckpointMng> m_checkpoint_mng; //!< Checkpoint manager
  Ref<IPropertyMng> m_property_mng; //!< Property manager
  ITimeStats* m_time_stats; //!< Execution time statistics
  ScopedPtrT<ITimeLoopMng> m_time_loop_mng; //!< Time loop manager
  ScopedPtrT<IServiceMng> m_service_mng; //!< Service manager
  ScopedPtrT<IPhysicalUnitSystem> m_physical_unit_system; //!< Physical unit system.
  String m_namespace_uri;
  String m_local_name;
  IModuleMaster* m_module_master; //!< Master module
  ScopedPtrT<ITimeHistoryMng> m_time_history_mng; //!< History manager
  ScopedPtrT<MeshMng> m_mesh_mng;
  MeshHandle m_default_mesh_handle;
  bool m_is_initialized; //!< \a true if initialized
  String m_case_full_file_name; //!< Case path
  String m_case_name; //!< Case name
  ByteUniqueArray m_case_bytes; //!< Case data.
  CaseOptionsMain* m_case_config; //!< Case config
  Directory m_export_directory; //!< Export directory
  Directory m_storage_directory; //!< Archive directory
  Directory m_listing_directory; //!< Listing directory
  Observable m_on_destroy_observable; //!< Observable upon destruction
  bool m_is_continue;
  IDirectExecution* m_direct_execution;
  ScopedPtrT<ILoadBalanceMng> m_lb_mng; //!< Load balancing characteristics manager
  ScopedPtrT<IConfiguration> m_configuration; //!< Configuration
  bool m_is_create_default_mesh_v2;
  ScopedPtrT<PropertyMngCheckpoint> m_property_mng_checkpoint;
  ScopedPtrT<LegacyMeshBuilder> m_legacy_mesh_builder;
  //! Indicates whether the service mechanism is used to read the mesh
  bool m_has_mesh_service = false;
  Ref<ICaseMeshMasterService> m_case_mesh_master_service;
  ObserverPool m_observers;
  Ref<IAcceleratorMng> m_accelerator_mng;

 private:

  void _doInitialPartition();
  void _doInitialPartitionForMesh(IMesh* mesh, const String& service_name);
  void _notifyWriteCheckpoint();
  void _printCPUAffinity();
  void _setDefaultAcceleratorDevice(Accelerator::AcceleratorRuntimeInitialisationInfo& config);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static ISubDomain* global_sub_domain = nullptr;

/* HP: ARCANE_IMPL_EXPORT pending finding a solution
 * to access traces outside of a service or module */
extern "C" ARCANE_IMPL_EXPORT ISubDomain* _arcaneGetDefaultSubDomain()
{
  return global_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ISubDomain*
arcaneCreateSubDomain(ISession* session, const SubDomainBuildInfo& sdbi)
{
  Ref<IParallelMng> pm = sdbi.parallelMng();
  Ref<IParallelMng> all_replica_pm = sdbi.allReplicaParallelMng();
  String case_file_name = sdbi.caseFileName();
  ByteConstArrayView bytes = sdbi.caseBytes();

  ITraceMng* tm = pm->traceMng();
  StringBuilder trace_id;
  trace_id += String::fromNumber(pm->commRank());
  if (all_replica_pm != pm) {
    trace_id += ",r";
    trace_id += pm->replication()->replicationRank();
  }
  trace_id += ",";
  trace_id += platform::getHostName();
  tm->setTraceId(trace_id.toString());

  auto* sd = new SubDomain(session, pm, all_replica_pm, case_file_name, bytes);
  sd->build();
  //GG: init is done by the caller
  //mng->initialize();
  global_sub_domain = sd;
  return sd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SubDomain::
SubDomain(ISession* session, Ref<IParallelMng> pm, Ref<IParallelMng> all_replica_pm,
          const String& case_file_name, ByteConstArrayView bytes)
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

// This call must be made from a critical section.
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
  m_mesh_mng = new MeshMng(m_application, m_variable_mng.get());
  m_default_mesh_handle = m_mesh_mng->createDefaultMeshHandle("Mesh0");
  m_service_mng = mf->createServiceMng(this);
  m_checkpoint_mng = mf->createCheckpointMng(this);
  m_module_mng = mf->createModuleMng(this);
  m_entry_point_mng = mf->createEntryPointMng(this);
  m_case_mng = mf->createCaseMng(this)->toReference();
  m_timer_mng = m_parallel_mng->timerMng();
  m_time_stats = m_parallel_mng->timeStats();
  m_time_loop_mng = mf->createTimeLoopMng(this);
  m_legacy_mesh_builder = new LegacyMeshBuilder(this, m_default_mesh_handle);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
initialize()
{
  namespace ax = Arcane::Accelerator;
  // Initialization of the parallel module
  // TODO: to be removed as it is no longer useful
  m_parallel_mng->initialize();

  {
    // Initializes the default runner based on the parameters
    // provided by the user.
    IApplication* app = application();
    ax::AcceleratorRuntimeInitialisationInfo config = app->acceleratorRuntimeInitialisationInfo();
    _setDefaultAcceleratorDevice(config);

    m_accelerator_mng->initialize(config);
    Runner runner = m_accelerator_mng->runner();
    const auto& device_info = runner.deviceInfo();
    info() << "DeviceInfo: name=" << device_info.name();
    info() << "DeviceInfo: description=" << device_info.description();

    if (isAcceleratorPolicy(runner.executionPolicy())) {
      m_parallel_mng->_internalApi()->setDefaultRunner(runner);
      m_all_replica_parallel_mng->_internalApi()->setDefaultRunner(runner);
    }
    m_variable_mng->_internalApi()->setAcceleratorMng(m_accelerator_mng);
  }

  _printCPUAffinity();

  IMainFactory* mf = m_application->mainFactory();

  m_time_loop_mng->build();

  m_has_mesh_service = false;

  // Reads the dataset.
  // This is only to ensure the syntactic validity
  // of the dataset. Options are not read.
  // This must be done before mesh creation
  // because the mesh instance depends on the dataset.
  ICaseDocument* case_document = nullptr;
  if (!m_case_bytes.empty()) {
    case_document = m_case_mng->readCaseDocument(m_case_full_file_name, m_case_bytes);
    if (!case_document)
      ARCANE_FATAL("Can not read case options");
    // Adds to the configuration those present in the dataset.
    {
      ConfigurationReader cr(traceMng(), m_configuration.get());
      cr.addValuesFromXmlNode(case_document->configurationElement(), ConfigurationReader::P_CaseDocument);
    }
  }

  properties::readFromConfiguration(m_configuration.get(), *this);

  if (case_document) {
    // Checks if there is a <meshes> tag indicating that the creation and
    // reading of the mesh will be managed by a service
    XmlNode meshes_elem = case_document->meshesElement();
    if (!meshes_elem.null()) {
      info() << "Using mesh service to create and allocate meshes";
      m_has_mesh_service = true;
    }
    // TODO: in this case, check that we do not have the historical 'mesh' element.
  }
  // Creates the default mesh. It must be done before the creation
  // of services, as the latter may need it.
  if (!m_has_mesh_service)
    if (!m_is_create_default_mesh_v2)
      m_legacy_mesh_builder->createDefaultMesh();

  ScopedPtrT<IServiceLoader> service_loader(mf->createServiceLoader());
  service_loader->loadSubDomainServices(this);

  // The master module must always be created before the others so that
  // its entry points are called first and last
  m_module_master = mf->createModuleMaster(this);
  m_time_history_mng = mf->createTimeHistoryMng(this);

  service_loader->initializeModuleFactories(this);
  m_lb_mng = mf->createLoadBalanceMng(this);

  m_property_mng_checkpoint = new PropertyMngCheckpoint(this);

  m_observers.addObserver(this,
                          &SubDomain::_notifyWriteCheckpoint,
                          m_checkpoint_mng->writeObservable());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
_setDefaultAcceleratorDevice(Accelerator::AcceleratorRuntimeInitialisationInfo& config)
{
  // Choice of accelerator to use
  // If multiple accelerators are available, one must be chosen by default.
  // We assume that all nodes have the same number of accelerators.
  // In this case, we take the i-th available accelerator as the default,
  // with \a i chosen as our rank modulo the number of available accelerators on
  // the node.
  // NOTE: this works well if we use a single process per accelerator.
  // If we use more, we should rather take consecutive ranks for
  // the same accelerator.
  // TODO: Also look into how to rather use the accelerator closest
  // to our rank. For this, we would need to use libraries like HWLOC
  //
  auto* device_list = Runner::deviceInfoList(config.executionPolicy());

  Int32 modulo_device = 0;
  if (device_list) {
    Int32 nb_device = device_list->nbDevice();
    info() << "DeviceInfo: nb_device=" << nb_device;
    modulo_device = nb_device;
  }

  // TODO: do this elsewhere
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACCELERATOR_PARALLELMNG_RANK_FOR_DEVICE", true)) {
    Int32 modulo = v.value();
    if (modulo == 0)
      modulo = 1;
    modulo_device = modulo;
    info() << "Use commRank() to choose accelerator device with modulo=" << modulo;
  }
  if (modulo_device != 0) {
    Int32 device_rank = m_parallel_mng->commRank() % modulo_device;
    info() << "Using device number=" << device_rank;
    config.setDeviceId(Accelerator::DeviceId(device_rank));
  }
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

  platform::callDotNETGarbageCollector();

  // Normally we should be able to remove this test because there should no longer
  // be any references to services or modules. This is the case
  // with the 'coreclr' implementation but not with 'mono'. So we
  // leave this test for now.
  // Starting from version 3.7.8, most potential GC issues are
  // fixed, so it is not necessary to return directly. Nevertheless
  // there are still problems with the use of 'ICaseFunction'.
  // We therefore keep the possibility of changing the behavior if an environment
  // variable is set.
  if (m_application->hasGarbageCollector()) {
    bool do_return = true;
    String x = platform::getEnvironmentVariable("ARCANE_DOTNET_USE_LEGACY_DESTROY");
    if (x == "1")
      do_return = true;
    if (x == "0")
      do_return = false;
    if (do_return)
      return;
  }

  m_module_master = nullptr;

  m_module_mng->removeAllModules();
  m_service_mng = nullptr;
  m_time_history_mng = nullptr;
  m_mesh_mng->destroyMeshes();

  m_time_loop_mng = nullptr;
  m_case_mng.reset();
  m_entry_point_mng = nullptr;
  m_module_mng = nullptr;
  m_io_mng = nullptr;

  m_lb_mng = nullptr;
  m_property_mng.reset();

  // Since all objects may contain variables, we must be
  // certain to destroy the variable manager last.
  m_variable_mng->_internalApi()->removeAllVariables();
  m_variable_mng = nullptr;

  delete this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
dumpInfo(std::ostream& o)
{
  m_module_mng->dumpList(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh allocation.
 *
 * This method only handles the construction of IMesh instances.
 * Reading these is done in readOrReloadMeshes().
 */
void SubDomain::
allocateMeshes()
{
  info() << "SubDomain: Allocating meshes";
  MessagePassing::dumpDateAndMemoryUsage(parallelMng(), traceMng());

  Timer::Action ts_action2(this, "AllocateMesh");
  Trace::Setter mci(traceMng(), _msgClassName());

  if (m_has_mesh_service) {
    info() << "** Reading mesh from mesh service";
    const CaseNodeNames* cnn = caseMng()->caseDocument()->caseNodeNames();
    String default_service_name = "ArcaneCaseMeshMasterService";

    // NOTE: this object will be destroyed by caseMng()
    ICaseOptions* opt = new CaseOptions(caseMng(), cnn->meshes);
    ServiceBuilder<ICaseMeshMasterService> sb(application(), opt);
    Ref<ICaseMeshMasterService> mbm = sb.createReference(default_service_name);
    m_case_mesh_master_service = mbm;
    m_case_mng->_internalImpl()->internalReadOneOption(mbm->_options(), true);
    mbm->createMeshes();
  }
  else {
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
  info() << "SubDomain: read or reload meshes";
  MessagePassing::dumpDateAndMemoryUsage(parallelMng(), traceMng());
  logdate() << "Initialisation du code.";

  Integer nb_mesh = m_mesh_mng->meshes().size();
  info() << " nb_mesh_created=" << nb_mesh
         << " is_continue?=" << m_is_continue;

  A_INFO("Test: {1}", A_TR2("nb_mesh_created", nb_mesh), A_TR(m_is_continue));

  //info() << format4("Test: {1}",{A_PR2("nb_mesh_created",nb_mesh),A_PR(m_is_continue)});
  //info() << format5(A_PR2("nb_mesh_created",nb_mesh),A_PR(m_is_continue));

  // Checks if profiling is enabled during initialization
  IProfilingService* ps = nullptr;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_PROFILE_CREATE_MESH", true))
    if (v.value() != 0)
      ps = platform::getProfilingService();

  {
    ProfilingSentryWithInitialize ps_sentry(ps);
    ps_sentry.setPrintAtEnd(true);

    if (m_is_continue) {
      for (Integer z = 0; z < nb_mesh; ++z) {
        IPrimaryMesh* mesh = m_mesh_mng->getPrimaryMesh(z);
        mesh->reloadMesh();
      }
    }
    else {
      if (m_has_mesh_service) {
        m_case_mesh_master_service->allocateMeshes();
      }
      else
        m_legacy_mesh_builder->readMeshes();
    }
  }

  //warning() << "CODE CHANGED IN DEV-X10. NEED TEST";
  //mesh->computeTiedInterfaces(mbi.m_xml_node);
  //! AMR : done in factory. This method is removed from IMesh.
  //      mesh->readAmrActivator(mbi.m_xml_node);

  // Checks mesh consistency
  for (Integer z = 0; z < nb_mesh; ++z) {
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
  if (!m_is_continue) {
    IInitialPartitioner* init_part = m_legacy_mesh_builder->m_initial_partitioner.get();
    if (init_part) {
      info() << "Using custom initial partitioner";
      init_part->partitionAndDistributeMeshes(m_mesh_mng->meshes());
    }
    else {
      if (m_case_mesh_master_service.get()) {
        m_case_mesh_master_service->partitionMeshes();
        m_case_mesh_master_service->applyAdditionalOperationsOnMeshes();
      }
      else if (m_legacy_mesh_builder->m_use_internal_mesh_partitioner)
        _doInitialPartition();
    }
  }

  // Display mesh information
  for (IMesh* mesh : m_mesh_mng->meshes()) {
    ScopedPtrT<IMeshStats> mh(IMeshStats::create(traceMng(), mesh, parallelMng()));
    mh->dumpStats();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calls the module initialization entry points.
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
  // NOTE: This code is only used by the historical mesh creation mechanism.
  // When meshes are used for services, the corresponding code is in ArcaneCaseMeshService.

  String test_service = "MeshPartitionerTester";

  for (IMesh* mesh : m_mesh_mng->meshes()) {
    bool is_mesh_allocated = mesh->isAllocated();
    info() << "InitialPartitioning mesh=" << mesh->name() << " is_allocated?=" << is_mesh_allocated;
    if (!is_mesh_allocated)
      continue;
    // Since 'parmetis' does not like repartitioning a mesh if one of the sub-domains
    // is empty (which is the case if only one processor created the meshes),
    // we must first use the basic partitioner which distributes the meshes
    // among the sub-domains, and then 'parmetis' if it is present
    if (m_legacy_mesh_builder->m_use_partitioner_tester) {
      Int64 nb_cell = mesh->nbCell();
      Int64 min_nb_cell = parallelMng()->reduce(Parallel::ReduceMin, nb_cell);
      info() << "Min nb cell=" << min_nb_cell;
      if (min_nb_cell == 0)
        _doInitialPartitionForMesh(mesh, test_service);
      else
        info() << "Mesh name=" << mesh->name() << " have cells. Do not use " << test_service;
    }
    else
      info() << "No basic partition first needed";
    _doInitialPartitionForMesh(mesh, m_legacy_mesh_builder->m_internal_partitioner_name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
_doInitialPartitionForMesh(IMesh* mesh, const String& service_name)
{
  info() << "DoInitialPartition. mesh=" << mesh->name() << " service=" << service_name;

  String lib_name = service_name;
  IMeshPartitionerBase* mesh_partitioner_base = nullptr;
  Ref<IMeshPartitionerBase> mesh_partitioner_base_ref;
  Ref<IMeshPartitioner> mesh_partitioner_ref;

  ServiceBuilder<IMeshPartitionerBase> sbuilder(this);
  mesh_partitioner_base_ref = sbuilder.createReference(service_name, mesh, SB_AllowNull);
  mesh_partitioner_base = mesh_partitioner_base_ref.get();

  if (!mesh_partitioner_base) {
    // If not found, search with the old interface 'IMeshPartitioner' for
    // compatibility reasons
    pwarning() << "No implementation for 'IMeshPartitionerBase' interface found. "
               << "Searching implementation for legacy 'IMeshPartitioner' interface";
    ServiceBuilder<IMeshPartitioner> sbuilder_legacy(this);
    mesh_partitioner_ref = sbuilder_legacy.createReference(service_name, mesh, SB_AllowNull);
    if (mesh_partitioner_ref.get())
      mesh_partitioner_base = mesh_partitioner_ref.get();
  }

  if (!mesh_partitioner_base) {
    // If not found, retrieve the list of possible values and display them.
    StringUniqueArray valid_names;
    sbuilder.getServicesNames(valid_names);
    String valid_values = String::join(", ", valid_names);
    String msg = String::format("The specified service for the initial mesh partitionment ({0}) "
                                "is not available (valid_values={1}). This service has to implement "
                                "interface Arcane::IMeshPartitionerBase",
                                lib_name, valid_values);
    ARCANE_THROW(ParallelFatalErrorException, msg);
  }

  bool is_dynamic = mesh->isDynamic();
  mesh->modifier()->setDynamic(true);
  mesh->utilities()->partitionAndExchangeMeshWithReplication(mesh_partitioner_base, true);
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

  // Save history manager data if requested
  if (caseOptionsMain()->doTimeHistory())
    m_time_history_mng->dumpHistory(true);

  m_time_loop_mng->execExitEntryPoints();

  m_parallel_mng->printStats();

  traceMng()->flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
checkId(const String& where, const String& id)
{
  Int64 len = id.length();
  const char* str = id.localstr();
  if (len == 0 || !str)
    throw BadIDException(where, id);

  if (!isalpha(str[0]))
    throw BadIDException(where, id);
  for (Int64 i = 1; i < len; ++i)
    if (!isalpha(str[i]) && !isdigit(str[i]) && str[i] != '_' && str[i] != '.' && str[i] != '-')
      throw BadIDException(where, id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
readCaseMeshes()
{
  Trace::Setter mci(traceMng(), _msgClassName());

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
  // Info about the time loop used.
  {
    ITimeLoopMng* tlm = timeLoopMng();
    ITimeLoop* time_loop = tlm->usedTimeLoop();
    XmlElement time_loop_elem(root, "timeloopinfo");
    XmlElement title(time_loop_elem, "title", time_loop->title());
    XmlElement description(time_loop_elem, "description", time_loop->description());

    String ustr_userclass("userclass");
    for (StringCollection::Enumerator j(time_loop->userClasses()); ++j;)
      XmlElement elem(time_loop_elem, ustr_userclass, *j);
  }

  VariableRefList var_ref_list;
  XmlElement modules(root, "modules");

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

  // List of modules with the variables they use.
  for (ModuleCollection::Enumerator i(moduleMng()->modules()); ++i;) {
    XmlElement module_element(modules, ustr_module);
    module_element.setAttrValue(ustr_name, (*i)->name());
    bool is_activated = (*i)->used();
    module_element.setAttrValue(ustr_activated, is_activated ? ustr_true : ustr_false);
    var_ref_list.clear();
    variableMng()->variables(var_ref_list, *i);
    for (VariableRefList::Enumerator j(var_ref_list); ++j;) {
      XmlElement variable_element(module_element, ustr_variable_ref);
      variable_element.setAttrValue(ustr_ref, (*j)->name());
    }
  }

  // List of variables.
  XmlElement variables(root, "variables");
  VariableCollection var_prv_list = variableMng()->variables();
  for (VariableCollection::Enumerator j(var_prv_list); ++j;) {
    XmlElement elem(variables, ustr_variable);
    IVariable* var = *j;
    String dim = String::fromNumber(var->dimension());
    elem.setAttrValue(ustr_name, var->name());
    elem.setAttrValue(ustr_datatype, dataTypeName(var->dataType()));
    elem.setAttrValue(ustr_dimension, dim);
    elem.setAttrValue(ustr_kind, itemKindName(var->itemKind()));
  }

  // List of option blocks
  const ICaseMng* cm = caseMng();
  CaseOptionsCollection blocks = cm->blocks();
  XmlElement blocks_elem(root, "caseblocks");
  for (CaseOptionsCollection::Enumerator i(blocks); ++i;) {
    const ICaseOptions* block = *i;
    XmlElement block_elem(blocks_elem, ustr_caseblock);
    block_elem.setAttrValue(ustr_tagname, block->rootTagName());
    const IModule* block_module = block->caseModule();
    if (block_module)
      block_elem.setAttrValue(ustr_module, block_module->name());
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
findMesh(const String& name, bool throw_exception)
{
  return m_mesh_mng->findMesh(name, throw_exception);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
_notifyWriteCheckpoint()
{
  info(4) << "SubDomain::_notifyWriteCheckpoint()";
  {
    Properties time_stats_properties(propertyMng(), "TimeStats");
    timeStats()->saveTimeValues(&time_stats_properties);
  }
  {
    Properties p(propertyMng(), "MessagePassingStats");
    parallelMng()->stat()->saveValues(traceMng(), &p);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubDomain::
setIsInitialized()
{
  m_is_initialized = true;
  info(4) << "SubDomain::setIsInitialized()";
  {
    Properties time_stats_properties(propertyMng(), "TimeStats");
    timeStats()->mergeTimeValues(&time_stats_properties);
  }
  {
    Properties p(propertyMng(), "MessagePassingStats");
    parallelMng()->stat()->mergeValues(traceMng(), &p);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Displays the CPU affinity of all ranks.
 *
 * This is not active by default and is only used for debugging.
 */
void SubDomain::
_printCPUAffinity()
{
  bool do_print_affinity = false;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_PRINT_CPUAFFINITY", true))
    do_print_affinity = (v.value() > 0);
  if (!do_print_affinity)
    return;
  info() << "PrintCPUAffinity";
  IProcessorAffinityService* pas = platform::getProcessorAffinityService();

  // It is possible that some sub-domains do not have an instance
  // of 'IProcessorAffinityService'. In this case, only '0' will be displayed.

  UniqueArray<Byte> cpuset_bytes;
  const Int32 nb_byte = 48;
  if (pas) {
    String cpuset = pas->cpuSetString();
    cpuset_bytes = cpuset.bytes();
  }
  cpuset_bytes.resize(nb_byte, Byte{ 0 });
  cpuset_bytes[nb_byte - 1] = Byte{ 0 };

  IParallelMng* pm = parallelMng();
  const Int32 nb_rank = pm->commSize();
  bool is_master = pm->isMasterIO();

  UniqueArray2<Byte> all_cpuset_bytes;
  if (is_master)
    all_cpuset_bytes.resize(nb_rank, nb_byte);
  pm->gather(cpuset_bytes, all_cpuset_bytes.viewAsArray(), pm->masterIORank());
  if (is_master)
    for (Int32 i = 0; i < nb_rank; ++i) {
      info() << "CPUAffinity " << Trace::Width(5) << i << " = " << all_cpuset_bytes[i].data();
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename V> void SubDomain::
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

ARCANE_REGISTER_PROPERTY_CLASS(SubDomain, ());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
