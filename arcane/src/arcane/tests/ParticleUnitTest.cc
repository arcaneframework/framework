// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParticleUnitTest.cc                                         (C) 2000-2026 */
/*                                                                           */
/* Service de test de la gestion des particules.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceBuildInfo.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IParticleFamily.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemVector.h"
#include "arcane/core/IParticleExchanger.h"
#include "arcane/core/IAsyncParticleExchanger.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IExtraGhostParticlesBuilder.h"
#include "arcane/core/IndexedItemConnectivityView.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/ParticleUnitTest_axl.h"

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
 * \brief Test des particules
 */
class ParticleUnitTest
: public ArcaneParticleUnitTestObject
, public IExtraGhostParticlesBuilder
{
 public:

  explicit ParticleUnitTest(const ServiceBuildInfo& cb);
  ~ParticleUnitTest();

 public:
  
  void initializeTest() override;
  void executeTest() override;

  void computeExtraParticlesToSend() override;

  Int32ConstArrayView extraParticlesToSend(const String& family_name,Int32 sid) const override
  {
    if (family_name=="ArcaneParticlesWithGhost")
      return m_extra_ghost_particles_to_send[sid];
    else
      return Int32ConstArrayView();
  } ;


 private:

 private:
  
  IMesh* m_mesh;
  IItemFamily* m_particle_family;
  IItemFamily* m_particle_family_with_ghost;
  Int64 m_first_uid;
  SharedArray< SharedArray<Integer> > m_extra_ghost_particles_to_send;

 private:
  void _doTest(Integer iteration);
  void _doTest2(Integer iteration,bool allow_no_cell_particle);
  void _doTest3(Integer iteration);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParticleUnitTest::
ParticleUnitTest(const ServiceBuildInfo& sbi)
: ArcaneParticleUnitTestObject(sbi)
, m_mesh(sbi.mesh())
, m_particle_family(nullptr)
, m_particle_family_with_ghost(nullptr)
, m_first_uid(0)
{
  m_particle_family = sbi.mesh()->createItemFamily(IK_Particle,"ArcaneParticles");
  m_particle_family_with_ghost = sbi.mesh()->createItemFamily(IK_Particle,"ArcaneParticlesWithGhost");
  m_particle_family_with_ghost->toParticleFamily()->setEnableGhostItems(true) ;
  m_particle_temperature.fill(0);
  m_particle_temperature_with_ghost.fill(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParticleUnitTest::
~ParticleUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialisation du module lors du démarrage du cas.
 */
void ParticleUnitTest::
initializeTest()
{
  m_global_deltat = 0.1;
  m_first_uid = 0;

  VariableCellReal tcells(VariableBuildInfo(m_mesh,"TCell"));
  ENUMERATE_CELL(i_cell,allCells()){
    tcells[i_cell] = 1.;
  }
  //m_particle_energy.resize(5);
  info() << "Name of IParticleExchanger=" << options()->particleExchanger.name();
  IParticleExchanger* pe = options()->particleExchanger();
  pe->initialize(m_particle_family);

  mesh()->modifier()->addExtraGhostParticlesBuilder(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Prise en compte de la pseudo viscosité en fonction des options choisies
 */
void ParticleUnitTest::
executeTest()
{
  m_particle_family->setHasUniqueIdMap(false);

  Integer max_iteration = static_cast<Integer>(options()->maxIteration());
  // doTest2() et doTest3() partagent l'utilisation de m_first_uid.
  // TODO: utiliser une sous classe pour le test
  m_first_uid = 0;
  for( Integer i=0; i<max_iteration; ++i){
    _doTest2(i,false);
  }

  m_first_uid = 0;
  for( Integer i=0; i<max_iteration; ++i){
    _doTest3(i);
  }

  m_first_uid = 0;
  for( Integer i=0; i<max_iteration; ++i){
    _doTest2(i,true);
  }

  mesh()->modifier()->removeExtraGhostParticlesBuilder(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleUnitTest::
_doTest2(Integer iteration,bool allow_no_cell_particle)
{
  // Si \a allow_no_cell_particle est vrai, alors on indique que certaines
  // particules ne seront pas dans des mailles pour tester ce type de
  // configuration dans l'échangeur de particule.

  Int64 particle_per_cell = options()->nbParticlePerCell();
  if (iteration==0)
    particle_per_cell = options()->initNbParticlePerCell();
  info() << " TEST2 iteration=" << iteration << " increment=" << particle_per_cell
         << " nb_particle=" << m_particle_family->nbItem();
  Int64UniqueArray uids;

  IMesh* mesh = m_particle_family->mesh();
  IParallelMng* pm = m_particle_family->parallelMng();
  Int64 nb_own_cell = ownCells().size();
  Int64 max_own_cell = pm->reduce(Parallel::ReduceMax,nb_own_cell);
  Int32 comm_rank = pm->commRank();
  Int32 comm_size = pm->commSize();
  Int64 uid_increment = max_own_cell * particle_per_cell;
  Int64 first_uid = m_first_uid + uid_increment*comm_rank;
  ENUMERATE_CELL(icell,ownCells()){
    for( Integer i=0; i<particle_per_cell; ++i ){
      uids.add(first_uid);
      ++first_uid;
    }
  }

  m_first_uid = m_first_uid + uid_increment*comm_size;

  info() << "Create " << uids.size() << " particles";
  Int32UniqueArray particles_lid(uids.size());
  IParticleFamily* pf = m_particle_family->toParticleFamily();
  ParticleVectorView particles = pf->addParticles(uids,particles_lid);
  m_particle_family->endUpdate();

  // Récupère la liste des sous-domaines communiquants. Cela est utilisé
  // pour les particules qui ne sont pas dans les mailles.
  UniqueArray<Int32> cell_communicating_sub_domains;
  mesh->cellFamily()->getCommunicatingSubDomains(cell_communicating_sub_domains);
  Int32 nb_cell_communicating_sub_domain = cell_communicating_sub_domains.size();

  bool is_parallel = pm->isParallel();
  {
    // Détermine la liste des mailles sur la frontière (elles sont connectées par
    // une face à une maille d'un autre sous-domaine).
    // Note: normalement, cette liste ne change que si le maillage change.
    UniqueArray<Cell> boundary_cells;
    ENUMERATE_CELL(icell,ownCells()){
      Cell cell = *icell;
      for( Face face : cell.faces() ){
        Cell opposite_cell = face.oppositeCell(cell);
        if (opposite_cell.null())
          continue;
        if (!is_parallel || opposite_cell.owner()!=comm_rank){
          boundary_cells.add(cell);
          break;
        }
      }
    }
    Integer nb_boundary_cell = boundary_cells.size();
    if (nb_boundary_cell==0)
      ARCANE_FATAL("No cells on boundary");

    // Place les nouvelles particules dans ces mailles frontières.
    {
      Integer particle_index = 0;
      Integer nb_created_particle = uids.size();
      while (particle_index<nb_created_particle){
        for( Integer icell=0; icell<nb_boundary_cell; ++icell ){
          Particle p = Particle(particles[particle_index]);
          // Si on n'autorise les particules sans maille, indique qu'une
          // particule sur 4 est sans maille.
          if (allow_no_cell_particle && (icell%4)==0)
            pf->setParticleCell(p,Cell());
          else
            pf->setParticleCell(p,boundary_cells[icell]);
          m_particle_temperature[p] = 1.0;
          //m_particle_energy[p].fill(2.0);
          ++particle_index;
          if (particle_index>=nb_created_particle)
            break;
        }
      }
    }
    // Vérifie que la vue des connectités est correcte.
    {
      IndexedParticleCellConnectivityView particle_cell(pf);
      ENUMERATE_(Particle, ipart, m_particle_family->allItems()){
        Particle p = *ipart;
        bool has_cell1 = p.hasCell();
        bool has_cell2 = particle_cell.hasCell(ipart);
        if (has_cell1 != has_cell2)
          ARCANE_FATAL("Bad hasCell() for particle={0} has_cell1={1} has_cell2={2}", p, has_cell1, has_cell2);
        CellLocalId cell1 = p.cellId();
        CellLocalId cell2 = particle_cell.cellId(ipart);
        if (cell1 != cell2)
          ARCANE_FATAL("Bad cellId() for particle={0} cell1={1} cell2={2}", p, cell1, cell2);
      }
    }
  }

  Int32UniqueArray particles_local_id;
  if (is_parallel){
    IParticleExchanger* pe = options()->particleExchanger();

    Int32UniqueArray particles_sub_domain_to_send;
    Int32UniqueArray incoming_particles_local_id;
    ParticleVectorView particles_view = m_particle_family->allItems().view();
    Integer nb_local_particle = particles_view.size();
    info() << "LocalNbParticle to track = " << nb_local_particle;
    pe->beginNewExchange(nb_local_particle);

    bool is_finished = false;
    Int32 sub_iteration = 0;
    while (!is_finished){
      ++sub_iteration;
      Integer nb_remaining_particle = particles_view.size();
      particles_local_id.clear();
      particles_sub_domain_to_send.clear();
      Integer nb_particle_tracking_finished = 0;
      if (nb_remaining_particle>=0){
        Integer index = 0;

        ENUMERATE_(Particle, ipart, particles_view){
          ++index;
          Particle p = *ipart;
          if (nb_remaining_particle<10)
            info() << "PARTICLE=" << index << " " << ItemPrinter(p);
          Cell cell = p.cellOrNull();
          Int32 new_owner = A_NULL_RANK;
          if (!cell.null()){
            Integer nb_face = cell.nbFace();
            Face face = cell.face(index%nb_face);
            Cell opposite_cell = face.oppositeCell(cell);
            if (opposite_cell.null()){
              ++nb_particle_tracking_finished;
              continue;
            }
            if (opposite_cell.owner()==comm_rank){
              ++nb_particle_tracking_finished;
              continue;
            }
            new_owner = opposite_cell.owner();
          }
          else{
            if (index<10 || index<(nb_remaining_particle/2)){
              ++nb_particle_tracking_finished;
              continue;
            }
            // La particule n'a pas de maille, on l'envoie dans un sous-domaine
            // au hasard.
            Int64 puid = p.uniqueId();
            Int32 idx = static_cast<Int32>((puid + iteration) % nb_cell_communicating_sub_domain);
            new_owner = cell_communicating_sub_domains[idx];
          }
          particles_local_id.add(p.localId());
          particles_sub_domain_to_send.add(new_owner);
        }
      }

      incoming_particles_local_id.clear();
      IAsyncParticleExchanger* ae = pe->asyncParticleExchanger();

      if (ae){
        is_finished = ae->exchangeItemsAsync(nb_particle_tracking_finished,
                                             particles_local_id,particles_sub_domain_to_send,
                                             &incoming_particles_local_id, 0,
                                             nb_remaining_particle!=0);
      }
      else{
        info() << "** Iteration iter=" << iteration << " sub_iter=" << sub_iteration
               << " Particles to exchange n=" << particles_local_id.size()
               << " nb_remaining=" << nb_remaining_particle
               << " nb_finished=" << nb_particle_tracking_finished;

        is_finished = pe->exchangeItems(nb_particle_tracking_finished,
                                        particles_local_id,particles_sub_domain_to_send,
                                        &incoming_particles_local_id,0);
        info() << "Nb Particules: " << m_particle_family->nbItem()
               << " incoming=" << incoming_particles_local_id.size();
      }
      particles_view = m_particle_family->view(incoming_particles_local_id);
    }
  }
  pm->barrier();
  {
    // Supprime la moitié des particules créées
    particles_local_id.clear();
    ParticleGroup all_particles(m_particle_family->allItems());
    Integer nb_particle = all_particles.size();
    Integer nb_particle_to_remove = 0;
    Integer max_particle_to_remove = (Integer)(nb_particle * options()->destroyRatio());
    ENUMERATE_PARTICLE(i_part,all_particles){
      Particle part = *i_part;
      particles_local_id.add(part.localId());
      ++nb_particle_to_remove;
      if (nb_particle_to_remove>(max_particle_to_remove))
        break;
    }
    if (nb_particle_to_remove!=0){
      info() << "Supprime " << nb_particle_to_remove << " particule(s)";
      m_particle_family->toParticleFamily()->removeParticles(particles_local_id);
    }
    m_particle_family->endUpdate();
  }

  // Compacte toute les 2 itérations pour tester le compactage
  if ((iteration % 2) == 0){
    info() << "Compacting particle family";
    m_particle_family->compactItems(true);
    info() << "MemoryUsed = " << platform::getMemoryUsed();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleUnitTest::
_doTest3(Integer iteration)
{
  Int64 particle_per_cell = options()->nbParticlePerCell();
  //if (iteration==0)
  //  particle_per_cell = options()->initNbParticlePerCell();
  info() << " TEST3 iteration=" << iteration << " increment=" << particle_per_cell
         << " nb_particle=" << m_particle_family_with_ghost->nbItem();
  Int64UniqueArray uids;
  Int32UniqueArray cell_lids;

  IParallelMng* pm = subDomain()->parallelMng();
  Int64 nb_own_cell = ownCells().size();
  Int64 max_own_cell = pm->reduce(Parallel::ReduceMax,nb_own_cell);
  Int32 comm_rank = pm->commRank();
  Int32 comm_size = pm->commSize();
  Int64 uid_increment = max_own_cell * particle_per_cell;
  Int64 first_uid = m_first_uid + uid_increment*comm_rank;
  ENUMERATE_CELL(icell,ownCells()){
    for( Integer i=0; i<particle_per_cell; ++i ){
      uids.add(first_uid);
      cell_lids.add(icell->localId()) ;
      ++first_uid;
    }
  }
  // TODO: AJOUTER std::map<Particle.uid(),cell.uid> pour verifier apres changement que les cells des particules sont correctes.
  m_first_uid = m_first_uid + uid_increment*comm_size;

  info() << "Create " << uids.size() << " particles";
  Int32UniqueArray particles_lid(uids.size());
  IParticleFamily* pf = m_particle_family_with_ghost->toParticleFamily();
  pf->addParticles(uids,cell_lids,particles_lid);
  m_particle_family_with_ghost->endUpdate();

  // Initialize added particles
  ENUMERATE_PARTICLE(ipart,m_particle_family_with_ghost->view(particles_lid))
  {
    m_particle_temperature_with_ghost[ipart] = 0;
  }

  mesh()->modifier()->setDynamic(true);
  mesh()->modifier()->endUpdate() ;
  mesh()->updateGhostLayers(false);
  //mesh()->modifier()->endUpdate(true,false) ;

  info() << "Nb Cells :"<<allCells().size()<<" own : "<<ownCells().size() ;
  info() << "Nb Particules Family With Ghost: " << m_particle_family_with_ghost->nbItem()
         <<" own size : "<<m_particle_family_with_ghost->allItems().own().size();

  ENUMERATE_PARTICLE(i_part,m_particle_family_with_ghost->allItems().own()){
    m_particle_temperature_with_ghost[i_part] = m_particle_temperature_with_ghost[i_part] + 1.0;
  }
  m_particle_temperature_with_ghost.synchronize() ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleUnitTest::
computeExtraParticlesToSend()
{
  IParallelMng* pm = subDomain()->parallelMng();
  Int32 comm_rank = pm->commRank();
  Int32 comm_size = pm->commSize();
  m_extra_ghost_particles_to_send.resize(comm_size) ;
  for(Integer i=0;i<comm_size;++i)
    m_extra_ghost_particles_to_send[i].clear() ;
  if(pm->isParallel()){
    std::map<Integer,std::set<Integer> > boundary_cells_neighbs;
    ENUMERATE_CELL(icell,ownCells()){
      Cell cell = *icell;
      for( Face face : cell.faces() ){
        Cell opposite_cell = face.oppositeCell(cell);
        if (opposite_cell.null())
          continue;
        if (opposite_cell.owner()!=comm_rank){
          boundary_cells_neighbs[cell.localId()].insert(opposite_cell.owner()) ;
          break;
        }
      }
    }

    ENUMERATE_PARTICLE(i_part,m_particle_family_with_ghost->allItems().own()){
      Int32 part_lid = i_part->localId() ;
      Int32 cell_lid = i_part->cell().localId() ;
      std::map<Integer,std::set<Integer> >::const_iterator iter = boundary_cells_neighbs.find(cell_lid) ;
      if(iter!=boundary_cells_neighbs.end()){
        for(std::set<Integer>::const_iterator i=iter->second.begin();i!=iter->second.end();++i){
          m_extra_ghost_particles_to_send[*i].add(part_lid) ;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PARTICLEUNITTEST(ParticleUnitTest,ParticleUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
