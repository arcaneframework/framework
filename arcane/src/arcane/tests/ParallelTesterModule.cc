// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelTesterModule.cc                                     (C) 2000-2024 */
/*                                                                           */
/* Module de test du parallèlisme.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/Event.h"

#include "arcane/core/MeshVariableInfo.h"
#include "arcane/core/EntryPoint.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ModuleFactory.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IGetVariablesValuesParallelOperation.h"
#include "arcane/core/ITransferValuesParallelOperation.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/SharedVariable.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/IParticleFamily.h"
#include "arcane/core/IExtraGhostParticlesBuilder.h"
#include "arcane/core/IItemFamilySerializeStep.h"
#include "arcane/core/IItemFamilyPolicyMng.h"
#include "arcane/core/VariableSynchronizerEventArgs.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IVariableSynchronizerMng.h"

#include "arcane/SerializeBuffer.h"

#include "arcane/tests/StdScalarMeshVariables.h"
#include "arcane/tests/StdArrayMeshVariables.h"
#include "arcane/tests/StdScalarVariables.h"
#include "arcane/tests/TypesParallelTester.h"
#include "arcane/tests/ParallelTester_axl.h"

#include "arcane/parallel/BitonicSortT.H"
#include "arcane/IParallelExchanger.h"
#include "arcane/ISerializeMessage.h"

#include "arcane/IApplication.h"
#include "arcane/IMainFactory.h"

#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour tester les familles de particule.
 */
class ParticleFamilyTester
: public TraceAccessor
, public IExtraGhostParticlesBuilder
{
 public:
  ParticleFamilyTester(IItemFamily* family)
  : TraceAccessor(family->traceMng()), m_family(family),
    m_values1(VariableBuildInfo(family,"Values1")), m_first_uid(450000)
  {
    if (m_family->toParticleFamily()->getEnableGhostItems())
      m_family->mesh()->modifier()->addExtraGhostParticlesBuilder(this);
  }
 public:
  void unregisterBuilder()
  {
    if (m_family->toParticleFamily()->getEnableGhostItems())
      m_family->mesh()->modifier()->removeExtraGhostParticlesBuilder(this);
  }
 public:
  Int32ConstArrayView extraParticlesToSend(const String& family_name,Int32 sid) const override
  {
    if (family_name==m_family->name() && m_family->toParticleFamily()->getEnableGhostItems())
      return m_extra_ghost_particles_to_send[sid];
    else
      return Int32ConstArrayView() ;
  }

  void computeExtraParticlesToSend() override
  {
    // NOTE GG: code recopié depuis ParticleUnitTest. a mutualiser.
    info() << "ComputeExtraParticlesToSend";
    IParallelMng* pm = m_family->parallelMng();
    Int32 comm_rank = pm->commRank();
    Int32 comm_size = pm->commSize();
    m_extra_ghost_particles_to_send.resize(comm_size) ;
    for(Integer i=0;i<comm_size;++i)
      m_extra_ghost_particles_to_send[i].clear() ;
    if(pm->isParallel()){
      CellGroup own_cells = m_family->mesh()->ownCells();
      std::map<Int32,std::set<Int32> > boundary_cells_neighbs;
      ENUMERATE_CELL(icell,own_cells){
        Cell cell = *icell;
        for( FaceEnumerator iface(cell.faces()); iface.hasNext(); ++iface ){
          Face face = *iface;
          Cell opposite_cell = face.oppositeCell(cell);
          if (opposite_cell.null())
            continue;
          if (opposite_cell.owner()!=comm_rank){
            boundary_cells_neighbs[cell.localId()].insert(opposite_cell.owner()) ;
            break;
          }
        }
      }

      ENUMERATE_PARTICLE(i_part,m_family->allItems().own()){
        Int32 part_lid = i_part->localId() ;
        Int32 cell_lid = i_part->cell().localId() ;
        auto iter = boundary_cells_neighbs.find(cell_lid) ;
        if (iter!=boundary_cells_neighbs.end()){
          for( Int32 sid : iter->second )
            m_extra_ghost_particles_to_send[sid].add(part_lid) ;
        }
      }
    }
    for( Integer i=0, n=m_extra_ghost_particles_to_send.size(); i<n; ++i ){
      info() << "Send rank=" << i << " nb_particle=" << m_extra_ghost_particles_to_send[i];
    }
  }

 public:
  void addParticles()
  {
    //TODO: mettre une option du JDD pour le choix de particle_per_cell
    //TODO: ne pas mettre le même nombre de particules dans chaque maille
    Integer particle_per_cell = 12;
    info() << " BuildParticleFamily increment=" << particle_per_cell
           << " nb_particle=" << m_family->nbItem();
    Int64UniqueArray uids;
    Int32UniqueArray cells_lid;
    IParallelMng* pm = m_family->parallelMng();
    CellGroup own_cells = m_family->mesh()->ownCells();
    Integer nb_own_cell = own_cells.size();
    Integer max_own_cell = pm->reduce(Parallel::ReduceMax,nb_own_cell);
    Integer comm_rank = pm->commRank();
    Integer comm_size = pm->commSize();
    Integer uid_increment = max_own_cell * particle_per_cell;
    Int64 first_uid = m_first_uid + uid_increment*comm_rank;
    ENUMERATE_CELL(icell,own_cells){
      for( Integer i=0; i<particle_per_cell; ++i ){
        uids.add(first_uid);
        cells_lid.add(icell.itemLocalId());
        ++first_uid;
      }
    }

    m_first_uid = m_first_uid + uid_increment*comm_size;

    info() << "Create " << uids.size() << " particles";
    Int32UniqueArray particles_lid(uids.size());
    IParticleFamily* pf = m_family->toParticleFamily();
    ParticleVectorView particles = pf->addParticles(uids,cells_lid,particles_lid);
    // Redimensionne la variable pour pouvoir initialiser ses valeurs.
    m_family->partialEndUpdateVariable(m_values1.variable());
    ENUMERATE_PARTICLE(ipart,particles){
      Particle particle = *ipart;
      m_values1[ipart] = (Real)(particle.uniqueId().asInt64());
    }
    m_family->endUpdate();
  }

 private:
  IItemFamily* m_family;
  VariableParticleReal m_values1;
  Int64 m_first_uid;
  SharedArray< SharedArray<Integer> > m_extra_ghost_particles_to_send;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelTesterSerializeStep
: public IItemFamilySerializeStep
{
 public:
  ParallelTesterSerializeStep(IItemFamily* item_family)
  : m_family(item_family), m_nb_called(0)
  {
  }
 public:
  void initialize() override {}
  void notifyAction(const NotifyActionArgs& args) override
  {
    ITraceMng* tm = m_family->traceMng();
    tm->info() << "NOTIFY_ACTION action=" << (int)args.action();
    ++m_nb_called;
  }
  void serialize(const ItemFamilySerializeArgs&) override {}
  void finalize() override
  {
    // Normalement il doit y avoir eu 4 appels à notifyAction()
    if (m_nb_called!=4)
      ARCANE_FATAL("Bad number of calls for notifyAction() n={0}",m_nb_called);
  }
  ePhase phase() const override { return IItemFamilySerializeStep::PH_Variable; }
  IItemFamily* family() const override { return m_family; }
 private:
  IItemFamily* m_family;
  Integer m_nb_called;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test du parallélisme dans Arcane.
 *
 * Ce module teste les points suivants:
 * - synchronisations
 * - methodes de IParallelMng
 * - accumulate()
 * - getVariableValues()
 */
class ParallelTesterModule
: public ArcaneParallelTesterObject
, public TypesParallelTester
, public IItemFamilySerializeStepFactory
{
 public:

  ParallelTesterModule(const ModuleBuildInfo& cb);
  ~ParallelTesterModule();

 public:
  
  static void staticInitialize(ISubDomain* sd);

 public:
	
  VersionInfo versionInfo() const override { return VersionInfo(0,0,2); }

 public:

  void testLoop();
  void testInit();
  void testBuild();
  void testExit();

  IItemFamilySerializeStep* createStep(IItemFamily* family) override
  {
    return new ParallelTesterSerializeStep(family);
  }

 private:

 private:

  StdScalarVariables m_scalars;
  StdScalarMeshVariables<Node> m_nodes;
  StdScalarMeshVariables<Face> m_faces;
  StdScalarMeshVariables<Cell> m_cells;
  StdArrayMeshVariables<Node> m_array_nodes;
  StdArrayMeshVariables<Face> m_array_faces;
  StdArrayMeshVariables<Cell> m_array_cells;

  VariableNodeReal m_nodes_sub_domain;
  VariableCellReal m_cells_sub_domain;
  VariableCellReal m_cell_real_values;
  VariableCellReal m_cells_nb_shared;
  VariableCellArrayReal m_cells_nb_shared_array;
  VariableCellArrayReal m_empty_cells_array;

  VariableFaceReal m_face_real_values;

  VariableCellReal m_accumulate_real;
  VariableCellReal3 m_accumulate_real3;
  VariableCellInteger m_accumulate_integer;

  VariableCellReal m_cell_value;
  VariableCellInteger m_cell_loop;

  UniqueArray<ParticleFamilyTester*> m_particle_family_testers;

  IMeshPartitioner* m_mesh_partitioner;

  CellGroup m_partial_cell_group;
  ScopedPtrT<PartialVariableCellReal> m_partial_cell_variable;

  Integer m_nb_test_synchronize;

  EventObserverPool m_observer_pool;

 private:

  void _testSynchronize();
  void _testPartialSynchronize();
  void _testMultiSynchronize();
  void _testPartialMultiSynchronize();
  void _testSameValuesOnAllReplica();
  void _testDifferentValuesOnAllReplica();
  void _testAccumulate();
  void _testGetVariableValues();
  void _testLoadBalance();
  void _testGhostItemsReduceOperation();
  void _testTransferValues();
  void _writeAccumulateInfos(std::ostream& ofile,eItemKind ik,const String& msg);
  void _doInit();
  void _checkEnd();
  void _testBitonicSort();
  void _testPartialVariables();
  void _initParticleFamily(IItemFamily* family);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(ParallelTesterModule,TestParallel);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelTesterModule::
ParallelTesterModule(const ModuleBuildInfo& mb)
: ArcaneParallelTesterObject(mb)
, m_scalars(mb.meshHandle(),"TestParallelScalars")
, m_nodes(mb.meshHandle(),"TestParallelNodes")
, m_faces(mb.meshHandle(),"TestParallelFaces")
, m_cells(mb.meshHandle(),"TestParallelCells")
, m_array_nodes(mb.meshHandle(),"TestCheckpointArrayNodes")
, m_array_faces(mb.meshHandle(),"TestCheckpointArrayFaces")
, m_array_cells(mb.meshHandle(),"TestCheckpointArrayCells")
, m_nodes_sub_domain(VariableBuildInfo(this,"TestParallelNodeSubDomains"))
, m_cells_sub_domain(VariableBuildInfo(this,"TestParallelCellSubDomains"))
, m_cell_real_values(VariableBuildInfo(this,"TestParallelCellRealValues"))
, m_cells_nb_shared(VariableBuildInfo(this,"TestParallelCellsNbShared"))
, m_cells_nb_shared_array(VariableBuildInfo(this,"TestParallelCellsNbSharedArray"))
, m_empty_cells_array(VariableBuildInfo(this,"TestEmptyCellArray"))
, m_face_real_values(VariableBuildInfo(this,"TestParallelFaceRealValues"))
, m_accumulate_real(VariableBuildInfo(this,"TestParallelAccumulateReal"))
, m_accumulate_real3(VariableBuildInfo(this,"TestParallelAccumulateReal3"))
, m_accumulate_integer(VariableBuildInfo(this,"TestParallelAccumulateInteger"))
, m_cell_value(VariableBuildInfo(this,"TestParallelCellValue"))
, m_cell_loop(VariableBuildInfo(this,"TestParallelCellLoop"))
, m_mesh_partitioner(nullptr)
, m_nb_test_synchronize(1)
{
  addEntryPoint(this,"TP_testBuild",
                &ParallelTesterModule::testBuild,
                IEntryPoint::WBuild);
  addEntryPoint(this,"TP_testInit",
                &ParallelTesterModule::testInit,
                IEntryPoint::WStartInit);
  addEntryPoint(this,"TP_testLoop",
                &ParallelTesterModule::testLoop);
  addEntryPoint(this,"TP_testExit",
                &ParallelTesterModule::testExit,
                IEntryPoint::WExit);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
staticInitialize(ISubDomain* sd)
{
  ITimeLoopMng* tlm = sd->timeLoopMng();
  {
    ITimeLoop* time_loop = tlm->createTimeLoop(String("TestParallel"));
    {
      List<TimeLoopEntryPointInfo> clist;
      clist.add(TimeLoopEntryPointInfo("TestParallel.TP_testBuild"));
      time_loop->setEntryPoints(String(ITimeLoop::WBuild),clist);
    }
    {
      List<TimeLoopEntryPointInfo> clist;
      clist.add(TimeLoopEntryPointInfo("TestParallel.TP_testInit"));
      time_loop->setEntryPoints(String(ITimeLoop::WInit),clist);
    }
    {
      List<TimeLoopEntryPointInfo> clist;
      clist.add(TimeLoopEntryPointInfo("TestParallel.TP_testLoop"));
      time_loop->setEntryPoints(String(ITimeLoop::WComputeLoop),clist);
    }
    {
      List<TimeLoopEntryPointInfo> clist;
      clist.add(TimeLoopEntryPointInfo("TestParallel.TP_testExit"));
      time_loop->setEntryPoints(String(ITimeLoop::WExit),clist);
    }
    {
      StringList clist;
      clist.add("TestParallel");
      time_loop->setRequiredModulesName(clist);
    }
    tlm->registerTimeLoop(time_loop);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelTesterModule::
~ParallelTesterModule()
{
  for( ParticleFamilyTester* p : m_particle_family_testers )
    delete p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_checkEnd()
{
  info() << "Test parallel " << " N = " << options()->nbIteration();
  Integer nb = options()->nbIteration();
  if (nb<=0)
    nb = 1;

  Integer n = static_cast<Integer>(nb);
  Integer current_iteration = m_global_iteration();
  if (current_iteration>n)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_doInit()
{
  info() << "Test parallel init !";

  m_global_deltat = 0.1;
  IMesh* mesh = defaultMesh();
  IItemFamily* cell_family = mesh->cellFamily();
  bool has_partitioner = options()->loadBalanceService.size()==1;
  if (has_partitioner)
    m_mesh_partitioner = options()->loadBalanceService[0];
  mesh->modifier()->setDynamic(true);

  cell_family->policyMng()->addSerializeStep(this);

  auto on_synchronize_handler = [&](const VariableSynchronizerEventArgs& args)
  {
    info() << " SYNCHRONIZE !!! var=" << args.variables()[0]->fullName() << " time=" << args.elapsedTime();
  };
  cell_family->allItemsSynchronizer()->onSynchronized().attach(m_observer_pool,on_synchronize_handler);
  auto on_synchronize_handler2 = [&](const VariableSynchronizerEventArgs& args)
  {
    info() << " SYNCHRONIZE GLOBAL !!! var=" << args.variables()[0]->fullName() << " time=" << args.elapsedTime();
  };
  mesh->variableMng()->synchronizerMng()->onSynchronized().attach(m_observer_pool,on_synchronize_handler2);

  m_cells_nb_shared_array.resize(7);

  {
    Int32UniqueArray local_ids;
    ENUMERATE_CELL(icell,allCells()){
      Cell cell = *icell;
      if ((cell.uniqueId().asInt64() % 3)==0)
        local_ids.add(cell.localId());
    }
    m_partial_cell_group = cell_family->createGroup("PARTIAL_GROUP",local_ids);
    VariableBuildInfo vbi(this,"PartialCellVariable",cell_family->name(),m_partial_cell_group.name());
    m_partial_cell_variable = new PartialVariableCellReal(vbi);
    ENUMERATE_CELL(icell,m_partial_cell_group){
      Cell cell = *icell;
      (*m_partial_cell_variable)[icell] = (Real)cell.uniqueId().asInt64() + 2.0;
    }
  }

  {
    // Créé plusieurs familles de particules avec différentes caractéristiques
    IItemFamily* pf1 = mesh->createItemFamily(IK_Particle,"Particle1");
    m_particle_family_testers.add(new ParticleFamilyTester(pf1));

    IItemFamily* pf2 = mesh->createItemFamily(IK_Particle,"Particle2NoMap");
//    pf2->setHasUniqueIdMap(false); // to see why this. Cannot work when USE_GRAPH_CONNECTIVITY_POLICY is on.
    m_particle_family_testers.add(new ParticleFamilyTester(pf2));

    IItemFamily* pf3 = mesh->createItemFamily(IK_Particle,"Particle3Ghost");
    pf3->toParticleFamily()->setEnableGhostItems(true) ;
    m_particle_family_testers.add(new ParticleFamilyTester(pf3));
  }
  for( ParticleFamilyTester* p : m_particle_family_testers )
    p->addParticles();
  //mesh->modifier()->endUpdate(true,false);
  mesh->modifier()->endUpdate();
  mesh->updateGhostLayers(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
testBuild()
{
  info() << "TEST BUILD";
  // Créé un autre maillage pour s'assurer que le partitonneur interne
  // fonctionne bien avec un 2ème maillage vide.
  ISubDomain* sd = subDomain();
  IApplication* app = sd->application();
  IPrimaryMesh* new_mesh = app->mainFactory()->createMesh(sd,"Mesh2");
  new_mesh->setDimension(2);
  // N'alloue pas le maillage pour vérifier que le partitionnement n'a pas lieu
  // si le maillage n'est pas alloué.
  //new_mesh->allocateCells(0,Int64ConstArrayView(),false);
  //new_mesh->endAllocate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
testInit()
{
  _doInit();
  m_nb_test_synchronize = options()->nbTestSync();
  m_nodes_sub_domain.fill(0.);
  {
    IMesh* mesh = defaultMesh();
    {
      ENUMERATE_NODE(i,mesh->ownNodes()){
        const Node& node = *i;
        m_nodes_sub_domain[*i] = node.owner();
      }
      m_nodes_sub_domain.synchronize();
    }
    {
      ENUMERATE_CELL(i,mesh->ownCells()){
        const Cell& cell = *i;
        m_cells_sub_domain[*i] = cell.owner();
      }
      m_cells_sub_domain.synchronize();
    }
    {
      ENUMERATE_CELL(icell,mesh->allCells()){
        const Cell& cell = *icell;
        m_cell_loop[icell] = cell.owner()*5 + 5;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
testLoop()
{
  _checkEnd();
  if (m_nb_test_synchronize>=1){
    _testSynchronize();
    _testPartialSynchronize();
    _testMultiSynchronize();
    _testPartialMultiSynchronize();
    _testSameValuesOnAllReplica();
    _testDifferentValuesOnAllReplica();
  }
  Timer timer(subDomain(),"ParallelTesterModule::testLoop",Timer::TimerReal);
  {
    Timer::Sentry sentry(&timer);
    switch(options()->testId){
    case TestAll:
      _testAccumulate();
      _testGhostItemsReduceOperation();
      _testBitonicSort();
      _testLoadBalance();
      _testGetVariableValues();
      _testGhostItemsReduceOperation();
      _testTransferValues();
      _testPartialVariables();
      _testAccumulate();
      break;
    case TestNone:
      break;
    case TestLoadBalance:
      _testLoadBalance();
      break;
    case TestGetVariableValues:
      _testGetVariableValues();
      break;
    case TestGhostItemsReduceOperation:
      _testGhostItemsReduceOperation();
      break;
    case TestTransferValues:
      _testTransferValues();
      break;
    }
  }
  _testPartialVariables();
  if (m_mesh_partitioner){
    info() << "Set mesh partitioner";
    subDomain()->timeLoopMng()->registerActionMeshPartition(m_mesh_partitioner);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
testExit()
{
  for( ParticleFamilyTester* p : m_particle_family_testers )
    p->unregisterBuilder();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testSynchronize()
{
  info() << "Test synchronize";
  {
    info() << "Begin create variable";
    VariableCellArrayReal cells(VariableBuildInfo(this,"Toto"));
    cells.resize(5);
    info() << "End create variable";
  }

  IMesh* mesh = defaultMesh();

  Integer current_iteration = m_global_iteration();

  info() << "Initialize ArrayNode nb_node=" << nbNode();
  m_array_nodes.initialize();
  info() << "Initialize ArrayFace nb_face=" << nbFace();
  m_array_faces.initialize();
  info() << "Initialize ArrayCell nb_cell=" << nbCell();
  m_array_cells.initialize();

  // Teste la synchronisation avec une variable vide
  m_empty_cells_array.synchronize();
  m_empty_cells_array.synchronize();

  // Positionne les valeurs
  {
    m_nodes.setValuesWithViews(current_iteration,mesh->ownNodes());
    m_faces.setValuesWithViews(current_iteration,mesh->ownFaces());
    m_cells.setValuesWithViews(current_iteration,mesh->ownCells());
    m_array_nodes.setValues(current_iteration,mesh->ownNodes());
    m_array_faces.setValues(current_iteration,mesh->ownFaces());
    m_array_cells.setValues(current_iteration,mesh->ownCells());
  }

  // Synchronise les valeurs
  for( Integer i=0; i<m_nb_test_synchronize; ++i ){
    m_nodes.synchronize();
    m_faces.synchronize();
    m_cells.synchronize();
    m_array_nodes.synchronize();
    m_array_faces.synchronize();
    m_array_cells.synchronize();
  }

  // Vérifie les valeurs
  {
    Integer nb_error = 0;

    nb_error += m_nodes.checkValues(current_iteration,mesh->allNodes());
    nb_error += m_faces.checkValues(current_iteration,mesh->allFaces());
    nb_error += m_cells.checkValues(current_iteration,mesh->allCells());
		info() << "NB ERROR SEQ=" << nb_error;
    nb_error += m_array_nodes.checkValues(current_iteration,mesh->allNodes());
    nb_error += m_array_faces.checkValues(current_iteration,mesh->allFaces());
    nb_error += m_array_cells.checkValues(current_iteration,mesh->allCells());
    if (nb_error!=0)
      ARCANE_FATAL("Error in synchronize test: n={0}",nb_error);
  }

  // Meme test en utilisant les vues
  {
    Integer iteration = current_iteration + 2;
    // Positionne les valeurs
    {
      m_nodes.setValues(iteration,mesh->ownNodes());
      m_faces.setValues(iteration,mesh->ownFaces());
      m_cells.setValues(iteration,mesh->ownCells());
    }

    // Synchronise les valeurs
    for( Integer i=0; i<m_nb_test_synchronize; ++i ){
      m_nodes.synchronize();
      m_faces.synchronize(); 
      m_cells.synchronize();
    }

    // Vérifie les valeurs
    {
      Integer nb_error = 0;

      nb_error += m_nodes.checkValues(iteration,mesh->allNodes());
      nb_error += m_faces.checkValues(iteration,mesh->allFaces());
      nb_error += m_cells.checkValues(iteration,mesh->allCells());
      info() << "NB ERROR_WITH_VIEW SEQ=" << nb_error;
      if (nb_error!=0)
        ARCANE_FATAL("Error in synchronize test: n={0}",nb_error);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testPartialSynchronize()
{
  IMesh* mesh = defaultMesh();
  Integer current_iteration = m_global_iteration();
  
  UniqueArray<Int32> even_cells;
  UniqueArray<Int32> even_nodes;
  UniqueArray<Int32> even_faces;
  UniqueArray<Int32> odd_cells;
  UniqueArray<Int32> odd_nodes;
  UniqueArray<Int32> odd_faces;
  
  m_array_cells.initialize();
  m_array_nodes.initialize();
  m_array_faces.initialize();
  
  VariableList cell_vars;
  m_cells.addToCollection(cell_vars);
  m_array_cells.addToCollection(cell_vars);
    
  VariableList node_vars;
  m_nodes.addToCollection(node_vars);
  m_array_nodes.addToCollection(node_vars);
  
  VariableList face_vars;
  m_array_faces.addToCollection(face_vars);
  m_faces.addToCollection(face_vars);
  
  
  ENUMERATE_CELL(i, allCells()) {
    Int64 uid = i->uniqueId();
    if (uid % 2 == 0) {
      even_cells.add(i->localId());
    } else {
      odd_cells.add(i->localId());
    }
  }
  
  ENUMERATE_NODE(i, allNodes()) {
    Int64 uid = i->uniqueId();
    if (uid % 2 == 0) {
      even_nodes.add(i->localId());
    } else {
      odd_nodes.add(i->localId());
    }
  }
  
  ENUMERATE_FACE(i, allFaces()) {
    Int64 uid = i->uniqueId();
    if (uid % 2 == 0) {
      even_faces.add(i->localId());
    } else {
      odd_faces.add(i->localId());
    }
  }
  
  // Synchronise les items d'UID pair
  
  m_cells.setEvenValues(current_iteration,mesh->ownCells());
  m_nodes.setEvenValues(current_iteration,mesh->ownNodes());
  m_faces.setEvenValues(current_iteration,mesh->ownFaces());
  m_array_cells.setEvenValues(current_iteration,mesh->ownCells());
  m_array_nodes.setEvenValues(current_iteration,mesh->ownNodes());
  m_array_faces.setEvenValues(current_iteration,mesh->ownFaces());

  for( Integer i=0; i<m_nb_test_synchronize; ++i ){
    cell_vars.each([&](IVariable* v){v->synchronize(even_cells);});
    node_vars.each([&](IVariable* v){v->synchronize(even_nodes);});
    face_vars.each([&](IVariable* v){v->synchronize(even_faces);});
  }
  
  // Synchronise les items d'UID impair
  
  m_cells.setOddValues(current_iteration,mesh->ownCells());
  m_nodes.setOddValues(current_iteration,mesh->ownNodes());
  m_faces.setOddValues(current_iteration,mesh->ownFaces());
  m_array_cells.setOddValues(current_iteration,mesh->ownCells());
  m_array_nodes.setOddValues(current_iteration,mesh->ownNodes());
  m_array_faces.setOddValues(current_iteration,mesh->ownFaces());
  
  for( Integer i=0; i<m_nb_test_synchronize; ++i ){
    cell_vars.each([&](IVariable* v){v->synchronize(odd_cells);});
    node_vars.each([&](IVariable* v){v->synchronize(odd_nodes);});
    face_vars.each([&](IVariable* v){v->synchronize(odd_faces);});
  }

  // Verifie
  
  {
    Integer nb_error = 0;

    nb_error += m_cells.checkGhostValuesOddOrEven(current_iteration,mesh->allCells());
    nb_error += m_nodes.checkGhostValuesOddOrEven(current_iteration,mesh->allNodes());
    nb_error += m_faces.checkGhostValuesOddOrEven(current_iteration,mesh->allFaces());
    
    info() << "NB ERROR SEQ=" << nb_error;
    nb_error += m_array_cells.checkGhostValuesOddOrEven(current_iteration,mesh->allCells());
    nb_error += m_array_nodes.checkGhostValuesOddOrEven(current_iteration,mesh->allNodes());
    nb_error += m_array_faces.checkGhostValuesOddOrEven(current_iteration,mesh->allFaces());

    if (nb_error!=0)
      ARCANE_FATAL("Error in partial synchronize test: n={0}",nb_error);
    
    info() << "PARTIAL SYNC OK.";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testPartialMultiSynchronize()
{
  IMesh* mesh = defaultMesh();
  Integer current_iteration = m_global_iteration();
  
  UniqueArray<Int32> even_cells;
  UniqueArray<Int32> even_nodes;
  UniqueArray<Int32> even_faces;
  UniqueArray<Int32> odd_cells;
  UniqueArray<Int32> odd_nodes;
  UniqueArray<Int32> odd_faces;
  
  m_array_cells.initialize();
  m_array_nodes.initialize();
  m_array_faces.initialize();
  
  VariableList cell_vars;
  m_cells.addToCollection(cell_vars);
  m_array_cells.addToCollection(cell_vars);
    
  VariableList node_vars;
  m_nodes.addToCollection(node_vars);
  m_array_nodes.addToCollection(node_vars);
  
  VariableList face_vars;
  m_array_faces.addToCollection(face_vars);
  m_faces.addToCollection(face_vars);
  
  
  ENUMERATE_CELL(i, allCells()) {
    Int64 uid = i->uniqueId();
    if (uid % 2 == 0) {
      even_cells.add(i->localId());
    } else {
      odd_cells.add(i->localId());
    }
  }
  
  ENUMERATE_NODE(i, allNodes()) {
    Int64 uid = i->uniqueId();
    if (uid % 2 == 0) {
      even_nodes.add(i->localId());
    } else {
      odd_nodes.add(i->localId());
    }
  }
  
  ENUMERATE_FACE(i, allFaces()) {
    Int64 uid = i->uniqueId();
    if (uid % 2 == 0) {
      even_faces.add(i->localId());
    } else {
      odd_faces.add(i->localId());
    }
  }
  
  // Synchronise les items d'UID pair
  
  m_cells.setEvenValues(current_iteration,mesh->ownCells());
  m_nodes.setEvenValues(current_iteration,mesh->ownNodes());
  m_faces.setEvenValues(current_iteration,mesh->ownFaces());
  m_array_cells.setEvenValues(current_iteration,mesh->ownCells());
  m_array_nodes.setEvenValues(current_iteration,mesh->ownNodes());
  m_array_faces.setEvenValues(current_iteration,mesh->ownFaces());

  for( Integer i=0; i<m_nb_test_synchronize; ++i ){
    mesh->cellFamily()->synchronize(cell_vars, even_cells);
    mesh->nodeFamily()->synchronize(node_vars, even_nodes);
    mesh->faceFamily()->synchronize(face_vars, even_faces);
  }
  
  // Synchronise les items d'UID impair
  
  m_cells.setOddValues(current_iteration,mesh->ownCells());
  m_nodes.setOddValues(current_iteration,mesh->ownNodes());
  m_faces.setOddValues(current_iteration,mesh->ownFaces());
  m_array_cells.setOddValues(current_iteration,mesh->ownCells());
  m_array_nodes.setOddValues(current_iteration,mesh->ownNodes());
  m_array_faces.setOddValues(current_iteration,mesh->ownFaces());
  
  for( Integer i=0; i<m_nb_test_synchronize; ++i ){
    mesh->cellFamily()->synchronize(cell_vars, odd_cells);
    mesh->nodeFamily()->synchronize(node_vars, odd_nodes);
    mesh->faceFamily()->synchronize(face_vars, odd_faces);
  }

  // Verifie
  
  {
    Integer nb_error = 0;

    nb_error += m_cells.checkGhostValuesOddOrEven(current_iteration,mesh->allCells());
    nb_error += m_nodes.checkGhostValuesOddOrEven(current_iteration,mesh->allNodes());
    nb_error += m_faces.checkGhostValuesOddOrEven(current_iteration,mesh->allFaces());
    
    info() << "NB ERROR SEQ=" << nb_error;
    nb_error += m_array_cells.checkGhostValuesOddOrEven(current_iteration,mesh->allCells());
    nb_error += m_array_nodes.checkGhostValuesOddOrEven(current_iteration,mesh->allNodes());
    nb_error += m_array_faces.checkGhostValuesOddOrEven(current_iteration,mesh->allFaces());

    if (nb_error!=0)
      ARCANE_FATAL("Error in partial synchronize test: n={0}",nb_error);
    
    info() << "PARTIAL MULTISYNC OK.";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testSameValuesOnAllReplica()
{
  info() << "Test if replica are synchronized";

  IMesh* mesh = defaultMesh();

  Integer current_iteration = m_global_iteration();

  info() << "Initialize ArrayCell nb_cell=" << nbCell();
  m_array_cells.initialize();

  // Positionne les valeurs
  {
    m_scalars.setValues(current_iteration);
    m_cells.setValues(current_iteration,mesh->allCells());
    m_array_cells.setValues(current_iteration,mesh->allCells());
  }
  {
    ENUMERATE_CELL(icell,mesh->allCells()){
      if (icell.index()>10)
        break;
      Cell cell = *icell;
      info() << "VALUE cell=" << cell.uniqueId() << " v=" << m_cells.m_real[icell];
    }
  }
  // Vérifie les valeurs
  {
    Integer nb_error = 0;

    nb_error += m_scalars.checkReplica();
		info() << "NB ERROR1=" << nb_error;
    nb_error += m_cells.checkReplica();
		info() << "NB ERROR2=" << nb_error;
    nb_error += m_array_cells.checkReplica();
		info() << "NB ERROR3=" << nb_error;
    if (nb_error!=0)
      fatal() << "Error in checkReplicaAreSynced test: n=" << nb_error;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Teste la comparaison de valeurs entre réplica dans le
 * cas où elles sont différentes.
 */
void ParallelTesterModule::
_testDifferentValuesOnAllReplica()
{
  IMesh* mesh = defaultMesh();
  IParallelReplication* pr = mesh->parallelMng()->replication();
  if (!pr->hasReplication())
    return;

  info() << "Test different values on replica";

  Integer current_iteration = m_global_iteration();

  info() << "Initialize ArrayCell nb_cell=" << nbCell();
  m_array_cells.initialize();

  Integer seed = current_iteration;

  CellGroup all_cells = mesh->allCells();
  // Positionne les valeurs identiques sur tout le maillage.
  {
    m_scalars.setValues(seed+1);
    m_cells.setValues(seed,all_cells);
    m_array_cells.setValues(seed,all_cells);
  }
  // Créé un sous-groupe et positionne sur ce sous-groupe
  // des valeurs différentes entre les réplica.
  // Normalement le nombre d'erreur par variable est
  // donc égal à la taille de ce groupe.
  CellGroup sub_group;
  {
    // Utilise une variable à laquelle on positionne à 1
    // les mailles qu'on souhaite ajouter au groupe.
    // Synchronize cette variable et créé le groupe correspondant
    // qui est garanti être le même entre tous les sous-domaines
    Integer index = 0;
    Integer next_index = 1;
    Int32UniqueArray local_ids;
    VariableCellInteger in_group(VariableBuildInfo(mesh,"InGroup"));
    in_group.fill(0);
    ENUMERATE_CELL(icell,all_cells){
      if (index>next_index){
        in_group[icell] = 1;
        next_index += index;
      }
      ++index;
    }
    in_group.synchronize();
    ENUMERATE_CELL(icell,all_cells){
      if (in_group[icell]==1)
        local_ids.add(icell.itemLocalId());
    }
    sub_group = mesh->cellFamily()->createGroup("SubGroup",local_ids,true);
    info() << "NB_IN_SUB_GROUP=" << sub_group.size();
  }
  // Positionne une valeurs différente sur \a sub_group.
  {
    seed += (1+pr->replicationRank());
    m_scalars.setValues(seed+1);
    m_cells.setValues(seed,sub_group);
    m_array_cells.setValues(seed,sub_group);
  }
  Integer nb_expected_error = 0;
  // Ajoute les erreurs pour les VariableScalar.
  // On compare tous les types de variable et il y en a 9.
  nb_expected_error += 9;
  // Ajoute les erreurs pour les VariableArray.
  // On compare tous les types de variable et il y en a 9 dans m_cells.
  nb_expected_error += sub_group.size() * 9;
  // Ajoutes les erreurs pour les VariableArray2.
  nb_expected_error += sub_group.size() * (m_array_cells.nbValuePerItem());

  {
    Integer nb_error = 0;

    nb_error += m_scalars.checkReplica();
		info() << "NB ERROR1=" << nb_error;
    nb_error += m_cells.checkReplica();
		info() << "NB ERROR2=" << nb_error;
    nb_error += m_array_cells.checkReplica();
		info() << "NB ERROR3=" << nb_error;
    info() << "CheckReplicaAreSynced test: n=" << nb_error << " expected=" << nb_expected_error;
    if (nb_error!=nb_expected_error)
      ARCANE_FATAL("Error in checkReplicaAreSynced test: n={0} expected={1}",
                   nb_error,nb_expected_error);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testMultiSynchronize()
{
  info() << "Test multi synchronize";

  IMesh* mesh = defaultMesh();

  Integer current_iteration = m_global_iteration();

  info() << "Initialize ArrayNode nb_node=" << nbNode();
  m_array_nodes.initialize();
  info() << "Initialize ArrayFace nb_face=" << nbFace();
  m_array_faces.initialize();
  info() << "Initialize ArrayCell nb_cell=" << nbCell();
  m_array_cells.initialize();

  Integer wanted_value = current_iteration + 1;
  // Positionne les valeurs
  {
    m_nodes.setValues(wanted_value,mesh->ownNodes());
    m_faces.setValues(wanted_value,mesh->ownFaces());
    m_cells.setValues(wanted_value,mesh->ownCells());
    m_array_nodes.setValues(wanted_value,mesh->ownNodes());
    m_array_faces.setValues(wanted_value,mesh->ownFaces());
    m_array_cells.setValues(wanted_value,mesh->ownCells());
  }

  // Synchronise les valeurs
  for( Integer i=0; i<m_nb_test_synchronize; ++i ){
    VariableList node_vars;
    m_nodes.addToCollection(node_vars);
    m_array_nodes.addToCollection(node_vars);
    mesh->nodeFamily()->synchronize(node_vars);

    VariableList face_vars;
    m_faces.addToCollection(face_vars);
    m_array_faces.addToCollection(face_vars);
    mesh->faceFamily()->synchronize(face_vars);

    VariableList cell_vars;
    m_cells.addToCollection(cell_vars);
    m_array_cells.addToCollection(cell_vars);
    mesh->cellFamily()->synchronize(cell_vars);
  }


  // Vérifie les valeurs
  {
    Integer nb_error = 0;

    nb_error += m_nodes.checkValues(wanted_value,mesh->allNodes());
    nb_error += m_faces.checkValues(wanted_value,mesh->allFaces());
    nb_error += m_cells.checkValues(wanted_value,mesh->allCells());
		info() << "NB ERROR SEQ=" << nb_error;
    nb_error += m_array_nodes.checkValues(wanted_value,mesh->allNodes());
    nb_error += m_array_faces.checkValues(wanted_value,mesh->allFaces());
    nb_error += m_array_cells.checkValues(wanted_value,mesh->allCells());
    if (nb_error!=0)
      fatal() << "Error in synchronize test: n=" << nb_error;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_writeAccumulateInfos(std::ostream& ofile,eItemKind ik,const String& msg)
{
  ARCANE_UNUSED(ik);
  //IMesh* mesh = subDomain()->defaultMesh();
  //VariableItemInteger& vsid = mesh->itemsSubDomainOwner(ik);
  //VariableArrayInteger& item_unique_id = mesh->itemsUniqueId(ik);
  //ItemInternalArrayView all_items(mesh->itemsInternal(ik));
  ENUMERATE_CELL(icell,allCells()){
    Cell cell = *icell;
    ofile << msg << " lid=" << cell.localId()
          << " UID " << cell.uniqueId() << " from " << cell.owner()
          << " Real: " << m_accumulate_real[cell]
          << " Real3: " << m_accumulate_real3[cell]
          << " Integer " << m_accumulate_integer[cell]
          << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testAccumulate()
{
  info() << "Test Accumulate\n";

  ISubDomain* sd = subDomain();
  IParallelMng* pm = sd->parallelMng();
  IMesh* mesh = sd->defaultMesh();
  eItemKind ik = IK_Cell; // Pour l'instant, test uniquement les mailles

  //VariableItemInteger& vsid = mesh->itemsSubDomainOwner(ik);
  //VariableArrayInteger& item_unique_id = mesh->itemsUniqueId(ik);
  Integer sid = pm->commRank();

  if (!pm->isParallel())
    return;

  // Au plus, 3 valeurs sont envoyées pour chaque éléments.
  //Integer nb_by_var = sid+1;
  //if (nb_by_var>3)
  //nb_by_var = 3;

  // Calcul le nombre d'éléments fantômes
  IItemFamily* cell_family = mesh->cellFamily();
  ItemGroup all_items = cell_family->allItems();
  //Integer nb_ghost = all_items.size() - all_items.own().size();
  //Integer acc_size = nb_ghost * nb_by_var;

  //m_accumulate_ids.resize(acc_size);
  //m_accumulate_real.resize(acc_size);
  //m_accumulate_real3.resize(acc_size);
  //m_accumulate_Integer.resize(acc_size);
  //m_accumulate_integer.resize(acc_size);

  m_accumulate_real.fill(0.0);
  m_accumulate_real3.fill(Real3::zero());
  m_accumulate_integer.fill(0);

  {
    //Integer counter = 0;
    ENUMERATE_CELL(i,all_items){
      Cell cell = *i;
      Int64 uid = cell.uniqueId();
      if (sid!=cell.owner()){
        Real r = static_cast<Real>(sid+1);
        Real3 r3(r,r+1.,r+2.);
        m_accumulate_real[cell] = r;
        m_accumulate_real3[cell] = r3;
        m_accumulate_integer[cell] = static_cast<Integer>(uid);
      }
    }
  }
  // On écrit  uniquement à la première itération
  bool need_write = m_global_iteration()==1;
  String output_file_name(options()->outputFile());
  if (!output_file_name.empty()){
    output_file_name = output_file_name + "-" + sid;
  }
  
  if (need_write && !output_file_name.empty()){
    std::ofstream ofile(output_file_name.localstr());
    _writeAccumulateInfos(ofile,ik,"Send");
  }

  //VariableList variables;
  //variables.add(m_accumulate_real.variable());
  //variables.add(m_accumulate_real3.variable());
  //variables.add(m_accumulate_Integer.variable());
  //variables.add(m_accumulate_integer.variable());
  //pm->accumulate(ik,m_accumulate_ids,variables);
  cell_family->reduceFromGhostItems(m_accumulate_real.variable(),Parallel::ReduceSum);
  if (need_write && !output_file_name.empty()){
    std::ofstream ofile(output_file_name.localstr(),std::ios::app);
    _writeAccumulateInfos(ofile,ik,"Recv");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testLoadBalance()
{
  // Test pour l'équilibrage de charge.
  // Pour avoir un cas déséquilibré, un sous-domaine effectue autant
  // de fois le calcul que son numéro de sous-domaine.
  info() << "Teste Equilibrage charge 2\n";

  IMesh* mesh = defaultMesh();

  SharedVariableNodeReal3 node_coord = mesh->sharedNodesCoordinates();
  Real total = 1.0;
  for( Integer z=0; z<options()->nbInternalLoop(); ++z ){
    ENUMERATE_CELL(icell,mesh->allCells()){
      const Cell& cell = *icell;
      for( Integer i=0, iz=m_cell_loop[cell]; i<iz; ++i ){
        for( NodeEnumerator inode(cell.nodes()); inode(); ++inode )
          z += Convert::toInteger(node_coord[inode].normL2());
      }
      m_cell_value[icell] = z;
      total += z;
    }
  }
  info() << "Total = " << total;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testGetVariableValues()
{
  info() << "Test IParallelMng::GetVariableValues\n";
  ISubDomain* sd = subDomain();
  IMesh* mesh = defaultMesh();
  IParallelMng* pm = sd->parallelMng();
  Integer sid = sd->subDomainId();
  Integer current_iteration = m_global_iteration();

  ItemGroup own_items(mesh->ownCells());
  Integer nb_own = own_items.size();
  Integer total_nb_item = pm->reduce(Parallel::ReduceSum,nb_own);
  Integer nb_send_item = sid*2 + (total_nb_item / 5);
  if (nb_send_item==0)
    nb_send_item = total_nb_item;

  // Détermine les unique_ids des entités dont on veux les valeurs
  UniqueArray<Int64> items_wanted_id(nb_send_item);
  for( Integer i=0; i<nb_send_item; ++i )
    items_wanted_id[i] = (i+current_iteration+(sid*nb_send_item+1)) % total_nb_item;

  VariableItemReal& var_values(m_cell_real_values);
  ENUMERATE_ITEM(i_item,own_items){
    const Item& item = *i_item;
    Int64 uid = item.uniqueId().asInt64();
    var_values[item] = Convert::toReal(uid + current_iteration);
  }
  var_values.synchronize();

  ENUMERATE_FACE(i_face,allFaces()){
    const Face& face = *i_face;
    Integer nb_cell = face.nbCell();
    Real v = 0.;
    for( Integer i=0; i<nb_cell; ++i )
      v += m_cell_real_values[face.cell(i)];
    v /= nb_cell;
    v += Convert::toReal(face.uniqueId().asInt64());
    m_face_real_values[i_face] = v;
  }

  RealUniqueArray output_values(nb_send_item);
  {
    auto op { ParallelMngUtils::createGetVariablesValuesOperationRef(pm) };
    op->getVariableValues(var_values,items_wanted_id,output_values);
  }

  // Maintenant, vérifie que la sortie est correcte
  {
    Integer nb_error = 0;
    for( Integer i=0; i<nb_send_item; ++i ){
      Int64 uid = items_wanted_id[i];
      Real expected_value = static_cast<Real>(uid+current_iteration);
      if (output_values[i]!=expected_value){
        if (nb_error<10)
          error() << "Values differents for uid=" << uid
                  << " value: " << output_values[i]
                  << " expected: " << expected_value;
        ++nb_error;
      }
    }
    if (nb_error!=0)
      fatal() << "Test IParallelMng::GetVariableValues() failed. "
              << nb_error << " error(s).";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testTransferValues()
{
  info() << "Test IParallelMng::TransfertValuesParallelOperation()\n";
  
  IParallelMng* pm = subDomain()->parallelMng();
  Integer nb_rank = pm->commSize();
  Integer my_rank = pm->commRank();
  Integer nb_send = nb_rank * 3 + pm->commRank() * 2;

  Int32UniqueArray send_ranks(nb_send);
  Int32UniqueArray send_int32_1(nb_send);
  Int32UniqueArray send_int32_2(nb_send);
  Int64UniqueArray send_int64(nb_send);
  RealUniqueArray send_real(nb_send);

  for( Integer i=0; i<nb_send; ++i ){
    Integer r = ((i*2) + my_rank) % nb_rank;
    if (r==my_rank)
      r = (r+1) % nb_rank;
    send_ranks[i] = r;
    send_int32_1[i] = my_rank + r;
    send_int32_2[i] = my_rank + 2*r;
    send_int64[i] = my_rank + 2*r;
    send_real[i] = (Real)(my_rank + 2*r);
  }

  // Calcule combien je dois recevoir de valeurs
  Integer nb_expected_recv = 0;
  UniqueArray<Int32> all_send_ranks;
  pm->allGatherVariable(send_ranks,all_send_ranks);
  for( Int32 x : all_send_ranks )
    if (x==my_rank)
      ++nb_expected_recv;

  SharedArray<Int32> recv_int32_1;
  SharedArray<Int32> recv_int32_2;
  SharedArray<Int64> recv_int64;
  SharedArray<Real> recv_real;

  {
    auto op { ParallelMngUtils::createTransferValuesOperationRef(pm) };
    op->setTransferRanks(send_ranks);
    op->addArray(send_int32_1,recv_int32_1);
    op->addArray(send_int32_2,recv_int32_2);
    op->addArray(send_int64,recv_int64);
    op->addArray(send_real,recv_real);
    op->transferValues();
  }

  Integer nb_error = 0;
  Integer recv_nb = recv_int32_1.size();
  info() << "** - ** NB RECEIVE = " << recv_nb << " expected=" << nb_expected_recv;
  if (recv_nb!=nb_expected_recv)
    ARCANE_FATAL("Bad number of received element n={0} expected={1}",recv_nb,nb_expected_recv);
  for( Integer i=0; i<recv_nb; ++i ){
    Int64 v32_1 = recv_int32_1[i];
    Int64 v32_2 = recv_int32_2[i];
    Int64 v64 = recv_int64[i];
    Real r = recv_real[i];
    if ((r!=(Real)v64) || (v32_2)!=(v64)){
      ++nb_error;
      if (nb_error<10)
        info() << " Bad value i32_1=" << v32_1 << " i32_2=" << v32_2 << " i64=" << v64
               << " r=" << r;
    }
  }
  if (nb_error!=0)
    fatal() << "Error in transfertValues() n=" << nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testGhostItemsReduceOperation()
{
  Real v = Real(m_global_iteration()+1);
  m_cells_nb_shared.fill(v);
  Integer n = m_cells_nb_shared_array.arraySize();
  ENUMERATE_CELL(icell,allCells()){
    for( Integer k=0; k<n; ++k )
      m_cells_nb_shared_array[icell][k] = v + (Real)(k+1);
  }
  info() << "Test GhostItemsReduceOperation";
  IItemFamily* family = mesh()->itemFamily(IK_Cell);
  family->reduceFromGhostItems(m_cells_nb_shared.variable(),Parallel::ReduceSum);
  family->reduceFromGhostItems(m_cells_nb_shared_array.variable(),Parallel::ReduceSum);

  ValueChecker vc(A_FUNCINFO);
  const bool is_debug = false;
  ENUMERATE_CELL(icell,allCells()){
    Real shared_value = m_cells_nb_shared[icell];
    Real nb_shared = shared_value / v;
    for( Integer k=0; k<n; ++k ){
      Real expected_value = (v + (Real)(k+1)) * nb_shared;
      if (is_debug)
        info() << "item=" << Cell(*icell).uniqueId() << " nb_shared=" << nb_shared << " k=" << k
               << " V=" << m_cells_nb_shared_array[icell][k] << " expected=" << expected_value;
      vc.areEqual(m_cells_nb_shared_array[icell][k],expected_value,"ReduceFromGhost with Array");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testBitonicSort()
{
  info() << "BITONIC SORT !!!";
  IMesh* mesh = defaultMesh();
  IParallelMng* pm = subDomain()->parallelMng();
  IItemFamily* family = mesh->cellFamily();
  VariableCellReal temperature(VariableBuildInfo(mesh,"Temperature"));
  ENUMERATE_CELL(icell,family->allItems()){
    Int64 uid = (*icell).uniqueId().asInt64();
    temperature[icell] = ((Real)uid) + 0.1;
  }

  Int64UniqueArray cells_uid;
  CellGroup own_group = family->allItems().own();
  Int32ConstArrayView own_group_local_ids = own_group.internal()->itemsLocalId();
  ENUMERATE_CELL(icell,own_group){
    Int64 uid = (*icell).uniqueId().asInt64();
    //cells_uid.add(uid+((uid*uid*uid)%10000));
    //info() << " ADD UID uid=" << uid;
    cells_uid.add(uid);
  }
  Parallel::BitonicSort<Int64> uid_sorter(pm);
  uid_sorter.sort(cells_uid);

  Int32ConstArrayView key_indexes = uid_sorter.keyIndexes();
  Int32ConstArrayView key_ranks = uid_sorter.keyRanks();
  Int64ConstArrayView keys = uid_sorter.keys();
  Int64 nb_item = keys.size();
  info() << "END SORT SIZE=" << nb_item << " KEY_SIZE=" << keys.size();
#if 0
  for( Integer i=0; i<math::min(nb_item,20); ++i ){
    info() << "I=" << i << " KEY=" << keys[i]
           << " INDEX=" << key_indexes[i]
           << " RANK=" << key_ranks[i];
  }
#endif
  UniqueArray< SharedArray<Int32> > indexes_to_send;
  SharedArray<Int32> ranks_to_send;
  UniqueArray< SharedArray<Int32> > indexes_to_recv;
  SharedArray<Int32> ranks_to_recv;
  {
    UniqueArray< SharedArray<Int32> > indexes_list(pm->commSize());
    UniqueArray< SharedArray<Int32> > own_indexes_list(pm->commSize());
    //Int32UniqueArray rank_to_sends;
    auto sd_exchange { ParallelMngUtils::createExchangerRef(pm) };
    for( Integer i=0; i<nb_item; ++i ){
      Int32 index = key_indexes[i];
      Int32 rank = key_ranks[i];
      if (indexes_list[rank].empty()){
        sd_exchange->addSender(rank);
      }
      indexes_list[rank].add(index);
      own_indexes_list[rank].add(i);
    }
    sd_exchange->initializeCommunicationsMessages();

    Int32ConstArrayView send_sd = sd_exchange->senderRanks();
    Integer nb_send = send_sd.size();
    indexes_to_recv.resize(nb_send);
    ranks_to_recv.resize(nb_send);
    for( Integer i=0; i<nb_send; ++i ){
      info() << " SEND TO: rank=" << send_sd[i];
      ISerializeMessage* send_msg = sd_exchange->messageToSend(i);
      Int32 dest_rank = send_sd[i];
      ISerializer* serializer = send_msg->serializer();
      Integer nb_to_send = indexes_list[dest_rank].size();
      indexes_to_recv[i] = own_indexes_list[dest_rank]; //indexes_list[dest_rank];
      ranks_to_recv[i] = dest_rank;
      serializer->setMode(ISerializer::ModeReserve);
      serializer->reserveInteger(1);
      serializer->reserveInt32(nb_to_send);
      serializer->allocateBuffer();
      serializer->setMode(ISerializer::ModePut);
      serializer->putInteger(nb_to_send);
      serializer->put(indexes_list[dest_rank]);
#if 0
      for( Integer z=0; z<nb_to_send; ++z ){
        Integer index = indexes_list[dest_rank][z];
        info() << " SEND Z=" << z << " RANK=" << dest_rank << " index=" << index
                << " own_index=" << indexes_to_recv[i][z];
      }
#endif
    }
    sd_exchange->processExchange();
    Int32ConstArrayView recv_sd = sd_exchange->receiverRanks();
    Integer nb_recv = recv_sd.size();
    indexes_to_send.resize(nb_recv);
    ranks_to_send.resize(nb_recv);
    for( Integer i=0; i<nb_recv; ++i ){
      info() << " RECEIVE FROM: rank=" << recv_sd[i];
      ISerializeMessage* recv_msg = sd_exchange->messageToReceive(i);
      Int32 orig_rank = recv_sd[i];
      ISerializer* serializer = recv_msg->serializer();
      serializer->setMode(ISerializer::ModeGet);
      Integer nb_to_recv = serializer->getInteger();
      SharedArray<Int32> recv_indexes = indexes_to_send[i]; 
      ranks_to_send[i] = orig_rank;
      recv_indexes.resize(nb_to_recv);
      serializer->get(recv_indexes);
      for( Integer z=0; z<nb_to_recv; ++z ){
        Integer index = recv_indexes[z];
        //info() << " RECV Z=" << z << " RANK=" << orig_rank << " index=" << index
        //     << " index2=" << own_group_local_ids[index];
        recv_indexes[z] = own_group_local_ids[index];
      }
    }

  }

  IData* data = temperature.variable()->data();
  {
    auto sd_exchange { ParallelMngUtils::createExchangerRef(pm) };

    for( Integer i=0, is=ranks_to_send.size(); i<is; ++i ){
      sd_exchange->addSender(ranks_to_send[i]);
    }
    Int32UniqueArray ranks_to_recv2;
    for( Integer i=0, is=ranks_to_recv.size(); i<is; ++i ){
      ranks_to_recv2.add(ranks_to_recv[i]);
    }
    sd_exchange->initializeCommunicationsMessages(ranks_to_recv2);
    Int32ConstArrayView send_sd = sd_exchange->senderRanks();
    Integer nb_send = send_sd.size();
    for( Integer i=0; i<nb_send; ++i ){
      info() << " SEND TO: rank=" << send_sd[i];
      ISerializeMessage* send_msg = sd_exchange->messageToSend(i);
      //Int32 dest_rank = send_sd[i];
      ISerializer* serializer = send_msg->serializer();
      serializer->setMode(ISerializer::ModeReserve);
      data->serialize(serializer,indexes_to_send[i],0);
      serializer->allocateBuffer();
      serializer->setMode(ISerializer::ModePut);
      data->serialize(serializer,indexes_to_send[i],0);
    }
    sd_exchange->processExchange();
    Int32ConstArrayView recv_sd = sd_exchange->receiverRanks();
    Integer nb_recv = recv_sd.size();
    indexes_to_send.resize(nb_recv);
    ranks_to_send.resize(nb_recv);
    Ref<IData> data2(data->cloneEmptyRef());
    //IData* data2 = data;
    Integer nb_item_as_integer = CheckedConvert::toInteger(nb_item);
    data2->resize(nb_item_as_integer);
    for( Integer i=0; i<nb_recv; ++i ){
      info() << " RECEIVE FROM: rank=" << recv_sd[i];
      ISerializeMessage* recv_msg = sd_exchange->messageToReceive(i);
      //Int32 orig_rank = recv_sd[i];
      ISerializer* serializer = recv_msg->serializer();
      serializer->setMode(ISerializer::ModeGet);
      data2->serialize(serializer,indexes_to_recv[i],0);
    }
    auto* true_data = dynamic_cast<IArrayDataT<Real>*>(data2.get());
    if (!true_data)
      ARCANE_FATAL("Bad Type");
    ConstArrayView<Real> true_array = true_data->view();
    {
      String fname(String("dump-")+pm->commRank());
      std::ofstream ofile(fname.localstr());
      for( Integer z=0, zs=nb_item_as_integer; z<zs; ++z )
        ofile << " VALUE Z=" << z << " v=" << true_array[z] << " key=" << keys[z] << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelTesterModule::
_testPartialVariables()
{
  Integer nb_error = 0;
  // Verifie que la variable partielle \a m_partial_cell_variable
  // a la même valeur que lors de _doInit().
  // Ce test est pertinent après un repartionnement de maillage par exemple.
  ENUMERATE_CELL(icell,m_partial_cell_group){
    Cell cell = *icell;
    Real wanted_value = (Real)cell.uniqueId().asInt64() + 2.0;
    Real value = (*m_partial_cell_variable)[icell];
    if (!math::isEqual(value,wanted_value)){
      ++nb_error;
      if (nb_error<10){
        info() << "Bad value uid=" << cell.uniqueId()
               << " wanted=" << wanted_value
               << " value=" << value;
      }
    }
    //info() << "V = " << (*m_partial_cell_variable)[icell];
  }
  if (nb_error!=0)
    ARCANE_FATAL("Bad values nb={0}",nb_error);
  info() << " TEST PARTIAL VARIABLES OK";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
