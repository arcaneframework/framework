// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckpointTesterService.cc                                  (C) 2000-2022 */
/*                                                                           */
/* Service de test des protections/reprises.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/OStringStream.h"

#include "arcane/BasicTimeLoopService.h"
#include "arcane/tests/StdArrayMeshVariables.h"
#include "arcane/tests/StdScalarMeshVariables.h"
#include "arcane/tests/CheckpointTester_axl.h"

#include "arcane/ITimeLoopMng.h"
#include "arcane/ICheckpointWriter.h"
#include "arcane/ICheckpointReader.h"
#include "arcane/IMesh.h"
#include "arcane/IVariableMng.h"
#include "arcane/IItemFamily.h"
#include "arcane/IParticleFamily.h"
#include "arcane/ItemVector.h"
#include "arcane/IParallelMng.h"
#include "arcane/IPropertyMng.h"
#include "arcane/ObserverPool.h"
#include "arcane/IItemConnectivityInfo.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/IMainFactory.h"
#include "arcane/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture de variables au format HDF5.
 */
class CheckpointTesterService
: public ArcaneCheckpointTesterObject
{
 public:

  CheckpointTesterService(const ServiceBuildInfo& sbi);
  ~CheckpointTesterService();

 public:

  //! Cette valeur doit correspondre à la période des sorties du JDD.
  const int CHECKPOINT_PERIOD = 3;

 public:

  void build() override {}
  void onTimeLoopStartInit() override;
  void onTimeLoopContinueInit() override;
  void onTimeLoopBeginLoop() override;
  void onTimeLoopRestore() override {}

 private:

  StdScalarMeshVariables<Node> m_nodes;
  StdScalarMeshVariables<Face> m_faces;
  StdScalarMeshVariables<Cell> m_cells;
  StdScalarMeshVariables<Particle>* m_particles;
  StdArrayMeshVariables<Node> m_array_nodes;
  StdArrayMeshVariables<Face> m_array_faces;
  StdArrayMeshVariables<Cell> m_array_cells;
  StdArrayMeshVariables<Particle>* m_array_particles;
  bool m_backward_done;
  Integer m_nb_iteration;
  Integer m_backward_iteration;
  bool m_is_continue;

  VariableScalarReal m_variable_no_restore;
  VariableScalarReal m_variable_with_property;
  VariableScalarString m_variable_scalar_string;
  VariableScalarString m_mesh_properties;
  VariableArrayString m_group_names;
  IItemFamily* m_particle_family;
  ObserverPool m_observer_pool;

 private:

  void _writeCheckpoint();
  void _compareCheckpoint();
  void _createParticlesVariables();
  void _createParticles();
  void _savePropertiesInVariable();
  String _getProperties();
  void _checkConnectivity();
  void _checkConnectivity(IItemFamily* family);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
static int full_property = IVariable::PNoNeedSync | IVariable::PNoRestore
 | IVariable::PSubDomainDepend | IVariable::PExecutionDepend;

CheckpointTesterService::
CheckpointTesterService(const ServiceBuildInfo& sbi)
: ArcaneCheckpointTesterObject(sbi)
, m_nodes(sbi.meshHandle(),"TestCheckpointNodes")
, m_faces(sbi.meshHandle(),"TestCheckpointFaces")
, m_cells(sbi.meshHandle(),"TestCheckpointCells")
, m_particles(0)
, m_array_nodes(sbi.meshHandle(),"TestCheckpointArrayNodes")
, m_array_faces(sbi.meshHandle(),"TestCheckpointArrayFaces")
, m_array_cells(sbi.meshHandle(),"TestCheckpointArrayCells")
, m_array_particles(0)
, m_backward_done(false)
, m_nb_iteration(0)
, m_backward_iteration(0)
, m_is_continue(false)
, m_variable_no_restore(VariableBuildInfo(sbi.subDomain(),"VariableNoRestore",
                                          IVariable::PNoRestore))
, m_variable_with_property(VariableBuildInfo(sbi.subDomain(),"VariableWithProperty",
                                             full_property))
, m_variable_scalar_string(VariableBuildInfo(sbi.subDomain(),"VariableString"))
, m_mesh_properties(VariableBuildInfo(sbi.meshHandle(),"TestCheckpointMeshProperties"))
, m_group_names({sbi.meshHandle(),"TestCheckpointGroupNames"})
, m_particle_family(nullptr)
{
  // Sauve les valeurs des propriétés dans une variable pour vérifier leur
  // relecture
  m_observer_pool.addObserver(this,
                              &CheckpointTesterService::_savePropertiesInVariable,
                              subDomain()->variableMng()->writeObservable());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CheckpointTesterService::
~CheckpointTesterService()
{
  delete m_particles;
  delete m_array_particles;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointTesterService::
_createParticlesVariables()
{
  String family_name = m_particle_family->name();
  m_particles = new StdScalarMeshVariables<Particle>(meshHandle(),"TestCheckpointParticles",
                                                     family_name);
  m_array_particles = new StdArrayMeshVariables<Particle>(meshHandle(),"TestCheckpointArrayParticles",
                                                          family_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointTesterService::
onTimeLoopStartInit()
{
  _checkConnectivity();
  m_global_deltat = 0.1;
  m_is_continue = false;
  m_particle_family = mesh()->findItemFamily(IK_Particle,"CheckpointParticle",true);
  m_particle_family->setHasUniqueIdMap(false);
  _createParticlesVariables();
  _createParticles();
  IntegerUniqueArray sizes(5);
  for( Integer i=0; i<5; ++i )
    sizes[i] = i;

  // Créé un maillage 'Mesh2' avec un sequentialParallelMng()
  // pour vérifier en reprise qu'il est bien recréé comme cela.
  {
    ISubDomain* sd = subDomain();
    IParallelMng* pm = sd->parallelMng();
    IParallelMng* seq_pm = pm->sequentialParallelMng();
    IPrimaryMesh* mesh2 = sd->application()->mainFactory()->createMesh(sd,seq_pm,"Mesh2");

    mesh2->setDimension(2);
    mesh2->allocateCells(0,Int64ConstArrayView(),false);
    mesh2->endAllocate();
  }

  {
    // Remplit 'm_group_names' avec la liste de tous les groupes existants
    ItemGroupCollection groups = mesh()->groups();
    Int32 nb_group = groups.count();
    m_group_names.resize(nb_group);
    info() << "NB_MESH_GROUP=" << nb_group;
    Int32 index = 0;
    for ( ItemGroupCollection::Enumerator igroup(groups); ++igroup; ++index ){
      ItemGroup g = *igroup;
      m_group_names[index] = g.name();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointTesterService::
onTimeLoopContinueInit()
{
  ISubDomain* sd = subDomain();

  m_is_continue = true;

  _checkConnectivity();

  // Sauve les valeurs actuelles des propriétés et vérifie qu'elles sont
  // identiques aux valeurs sauvées.
  info() << "OnTimeLoopContinueInit: propertyMng() notify write observers";
  sd->propertyMng()->writeObservable()->notifyAllObservers();
  String properties = _getProperties();

  if (properties!=m_mesh_properties()){
    fatal() << "Current properties and saved properties are different"
            << " saved=\n" << m_mesh_properties()
            << " current=\n" << properties;
  }

  m_particle_family = mesh()->findItemFamily(IK_Particle,"CheckpointParticle",true);
  bool has_map = m_particle_family->hasUniqueIdMap();
  info() << "Checkpoint HasUniqueIdMap property";
  if (has_map)
    fatal() << "family property 'map' is not handled";

  _createParticlesVariables();

  // Vérifie que les valeurs des variables sont correctes
  _compareCheckpoint();

  // Vérifie que les propriétés des variables sont bien conservées après une reprise
  {
    int p = m_variable_with_property.property();
    if (p!=full_property)
      fatal() << "variable properties not handled value="
              << p << " expected=" << full_property;
  }

  {
    int p = m_variable_no_restore.property();
    if (!(p & IVariable::PNoRestore))
      fatal() << "variable property 'PNoRestore' not handled";
  }

  // Vérifie que le maillage 'Mesh2' est bien créé en reprise
  // avec le sequentialParallelMng().
  IMesh* mesh2 = sd->findMesh("Mesh2");
  if (mesh2->parallelMng()!=sd->parallelMng()->sequentialParallelMng())
    ARCANE_FATAL("Mesh2 does not use sequentialParallelMng()");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointTesterService::
_checkConnectivity()
{
  _checkConnectivity(mesh()->cellFamily());
  _checkConnectivity(mesh()->faceFamily());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointTesterService::
_checkConnectivity(IItemFamily* family)
{
  IItemConnectivityInfo* family_local = family->localConnectivityInfos();
  IItemConnectivityInfo* family_global = family->globalConnectivityInfos();

  Integer nb_item =  family->nbItem();
  Integer nb_node_local = family_local->maxNodePerItem();
  Integer nb_node_global = family_global->maxNodePerItem();

  info() << "NB_ITEM=" << nb_item;
  info() << "NAX_NB_NODE family=" << family->name()
         << " global=" << nb_node_global << " local=" << nb_node_local;
  if (nb_item!=0 && nb_node_local==0)
    ARCANE_FATAL("Bad maxNodePerItem() nb_item={0} max_local={1}",nb_item,
                 nb_node_local);
  if (nb_node_global==0)
    ARCANE_FATAL("Bad maxNodePerItem() max_global={0}",nb_node_global);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointTesterService::
onTimeLoopBeginLoop()
{
  ITimeLoopMng* tm = subDomain()->timeLoopMng();
  ++m_nb_iteration;
  info() << "Test checkpoint " << " N = " << m_nb_iteration
         << " stop_reason=" << (int)tm->stopReason();
  Integer current_iteration = m_global_iteration();
  if (m_nb_iteration>options()->nbIteration()){
    tm->stopComputeLoop(false);
  }

  if ((current_iteration%CHECKPOINT_PERIOD)==0){
    _writeCheckpoint();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CheckpointTesterService::
_getProperties()
{
  OStringStream ostr;
  subDomain()->propertyMng()->print(ostr());
  return ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointTesterService::
_savePropertiesInVariable()
{
  subDomain()->propertyMng()->writeObservable()->notifyAllObservers();
  m_mesh_properties = _getProperties();
  info() << "PROPERTIES: " << m_mesh_properties();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointTesterService::
_writeCheckpoint()
{
  IMesh* mesh = this->mesh();
  Integer current_iteration = m_global_iteration();
  info() << "Writing values at iteration=" << current_iteration;

  m_variable_scalar_string = String("String_") + current_iteration;

  m_nodes.setValues(current_iteration, mesh->allNodes());
  m_faces.setValues(current_iteration, mesh->allFaces());
  m_cells.setValues(current_iteration, mesh->allCells());
  if (m_particles)
    m_particles->setValues(current_iteration,m_particle_family->allItems());
  m_array_nodes.setValues(current_iteration, mesh->allNodes());
  m_array_faces.setValues(current_iteration, mesh->allFaces());
  m_array_cells.setValues(current_iteration, mesh->allCells());
  if (m_array_particles)
    m_array_particles->setValues(current_iteration,m_particle_family->allItems());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointTesterService::
_compareCheckpoint()
{
  // Vérifie les valeurs
  Integer nb_error = 0;
  IMesh* mesh = this->mesh();
  Integer current_iteration = m_global_iteration();
  Integer saved_iteration = (current_iteration - (current_iteration % CHECKPOINT_PERIOD));
  info() << "Reading values at iteration=" << current_iteration
         << " saved_iteration=" << saved_iteration;

  nb_error += m_nodes.checkValues(saved_iteration,mesh->allNodes());
  nb_error += m_faces.checkValues(saved_iteration,mesh->allFaces());
  nb_error += m_cells.checkValues(saved_iteration,mesh->allCells());
  if (m_particles)
    nb_error += m_particles->checkValues(saved_iteration,m_particle_family->allItems());
  nb_error += m_array_nodes.checkValues(saved_iteration,mesh->allNodes());
  nb_error += m_array_faces.checkValues(saved_iteration,mesh->allFaces());
  nb_error += m_array_cells.checkValues(saved_iteration,mesh->allCells());
  if (m_array_particles)
    nb_error += m_array_particles->checkValues(saved_iteration,m_particle_family->allItems());

  {
    String scalar_string = String("String_") + saved_iteration;
    String value = m_variable_scalar_string.value();
    if (value!=scalar_string){
      info() << "Bad value for VariableScalarString"
             << " v='" << m_variable_scalar_string() << "'"
             << " expected='" << scalar_string << "'";

      info() << "Bad value for VariableScalarString"
             << " v='" << m_variable_scalar_string().utf8().size() << "'"
             << " expected='" << scalar_string.utf8().size() << "'";

      info() << "Bad value for VariableScalarString"
             << " v='" << m_variable_scalar_string().utf8() << "'"
             << " expected='" << scalar_string.utf8() << "'";

      cout << "InternalDump scalar_string=";
      scalar_string.internalDump(cout);
      cout << "\n";

      cout << "InternalDump variable_scalar_string=";
      value.internalDump(cout);
      cout << "\n";

      ++nb_error;
    }
  }
  if (nb_error!=0)
    fatal() << "Errors in checkValues(): " << nb_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CheckpointTesterService::
_createParticles()
{
  Int64UniqueArray uids;

  IParallelMng* pm = subDomain()->parallelMng();
  Integer particle_per_cell = 5;
  Integer base_first_uid = 153;
  Integer nb_own_cell = ownCells().size();
  Integer max_own_cell = pm->reduce(Parallel::ReduceMax,nb_own_cell);
  Int32 comm_rank = pm->commRank();
  //Integer comm_size = pm->commSize();
  Int64 uid_increment = ((Int64)max_own_cell) * ((Int64)particle_per_cell);
  Int64 first_uid = base_first_uid + uid_increment*comm_rank;
  ENUMERATE_CELL(icell,ownCells()){
    for( Integer i=0; i<particle_per_cell; ++i ){
      uids.add(first_uid);
      ++first_uid;
    }
  }

  //m_first_uid = base_first_uid + uid_increment*comm_size;

  info() << "Create " << uids.size() << " particles";
  //particles_lid.resize(uids.size());
  Int32UniqueArray particles_lid(uids.size());
  IParticleFamily* pfamily = m_particle_family->toParticleFamily();
  ParticleVectorView particles = pfamily->addParticles(uids,particles_lid);
  //ItemVectorView particles(m_particle_family->view(particles_lid));
  // Il faut affecter une maille à chaque particule
  {
    Integer index = 0;
    ENUMERATE_CELL(icell,ownCells()){
      for( Integer i=0; i<particle_per_cell; ++i ){
        pfamily->setParticleCell(particles[index],*icell);
        //uids.add(first_uid);
        //++first_uid;
        ++index;
      }
    }
  }

  m_particle_family->endUpdate();
  m_particle_family->checkValidConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_CHECKPOINTTESTER(CheckpointTesterService,
                                         CheckpointTesterService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
