// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorParticlesUnitTest.cc                             (C) 2000-2026 */
/*                                                                           */
/* Service de test des particules pour les accelerateurs.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IParticleFamily.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IndexedItemConnectivityView.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/VariableViews.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandEnumerate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la classe 'AcceleratorViews'.
 */
class AcceleratorParticlesUnitTest
: public BasicUnitTest
{
 public:

  explicit AcceleratorParticlesUnitTest(const ServiceBuildInfo& sbi);
  ~AcceleratorParticlesUnitTest() override;

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  Runner m_runner;
  RunQueue m_queue;
  IMesh* m_mesh = nullptr;
  IItemFamily* m_particle_family = nullptr;

 private:

  void _setCellArrayValue(Integer seed);
  void _checkCellArrayValue(const String& message) const;

  void _createParticles();

 public:

  void _executeTest1();
  void _executeTest2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AcceleratorParticlesUnitTest, IUnitTest,
                                           AcceleratorParticlesUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorParticlesUnitTest::
AcceleratorParticlesUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
, m_mesh(sb.mesh())
{
  m_particle_family = m_mesh->createItemFamily(IK_Particle, "ArcaneParticles");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorParticlesUnitTest::
~AcceleratorParticlesUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorParticlesUnitTest::
initializeTest()
{
  m_runner = subDomain()->acceleratorMng()->runner();
  m_queue = subDomain()->acceleratorMng()->queue();

  _createParticles();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorParticlesUnitTest::
_createParticles()
{
  // Create particles
  Int32 nb_particle_per_cell = 5;
  Int64 base_first_uid = 67;

  IParallelMng* pm = m_particle_family->parallelMng();
  Int32 nb_own_cell = ownCells().size();
  Int64 i64_nb_own_cell = nb_own_cell;
  Int64 max_own_cell = pm->reduce(Parallel::ReduceMax, i64_nb_own_cell);
  Int32 comm_rank = pm->commRank();
  Int64 uid_increment = max_own_cell * nb_particle_per_cell;
  Int64 first_uid = base_first_uid + uid_increment * comm_rank;
  Int32 nb_local_particle = nb_own_cell * nb_particle_per_cell;

  UniqueArray<Int64> particle_uids(nb_local_particle);
  UniqueArray<Int32> particle_cell_local_ids(nb_local_particle);

  {
    Int32 index = 0;
    ENUMERATE_ (Cell, icell, ownCells()) {
      Int32 cell_lid = icell.itemLocalId();
      for (Integer i = 0; i < nb_particle_per_cell; ++i) {
        particle_uids[index] = first_uid;
        ++first_uid;
        particle_cell_local_ids[index] = cell_lid;
        ++index;
      }
    }
  }

  info() << "Create " << nb_local_particle << " particles";
  UniqueArray<Int32> particles_lid(nb_local_particle);
  IParticleFamily* pf = m_particle_family->toParticleFamily();
  pf->addParticles(particle_uids, particle_cell_local_ids, particles_lid);
  m_particle_family->endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorParticlesUnitTest::
executeTest()
{
  _executeTest1();
  _executeTest2();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorParticlesUnitTest::
_executeTest1()
{
  info() << "Test1";
  Int32 max_id = m_particle_family->maxLocalId();
  NumArray<Int32, MDDim1> particle_cell_lids(max_id);
  VariableParticleInt32 var_particle_cell_lids(VariableBuildInfo(m_particle_family, "ParticleLocalId"));

  // Vérifie que l'accès via la connectivité donne la même valeur que
  // l'accès via l'entité
  {
    auto command = makeCommand(m_queue);
    auto out_particle_cell_lids = viewOut(command, particle_cell_lids);
    auto out_var_particle_cell_lids = viewOut(command, var_particle_cell_lids);
    IndexedParticleCellConnectivityView particle_cell_connectivity(m_particle_family);
    command << RUNCOMMAND_ENUMERATE (ParticleLocalId, particle_lid, m_particle_family->allItems())
    {
      out_particle_cell_lids[particle_lid] = particle_cell_connectivity.cellId(particle_lid);
      out_var_particle_cell_lids[particle_lid] = particle_cell_connectivity.cellId(particle_lid);
    };
  }

  ENUMERATE_ (Particle, ipart, m_particle_family->allItems()) {
    Particle p = *ipart;
    Int32 cell1_lid = p.cellId();
    Int32 cell2_lid = particle_cell_lids[p.localId()];
    Int32 cell3_lid = var_particle_cell_lids[p];
    if (cell1_lid != cell2_lid)
      ARCANE_FATAL("Incoherent cell for particle p={0} cell1_lid={1} cell2_lid={2}", p.localId(), cell1_lid, cell2_lid);
    if (cell1_lid != cell3_lid)
      ARCANE_FATAL("Incoherent cell for particle p={0} cell1_lid={1} cell3_lid={2}", p.localId(), cell1_lid, cell3_lid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorParticlesUnitTest::
_executeTest2()
{
  info() << "Test2 : check SetParticleCellId";
  Int32 max_id = m_particle_family->maxLocalId();
  Int32 max_cell_lid = m_mesh->cellFamily()->maxLocalId();
  NumArray<Int32, MDDim1> particle_cell_lids(max_id);
  VariableParticleInt32 var_particle_cell_lids(VariableBuildInfo(m_particle_family, "ParticleLocalId"));

  // Vérifie que l'accès via la connectivité donne la même valeur que
  // l'accès via l'entité
  {
    auto command = makeCommand(m_queue);
    auto out_particle_cell_lids = viewOut(command, particle_cell_lids);
    auto out_var_particle_cell_lids = viewOut(command, var_particle_cell_lids);
    MutableIndexedParticleCellConnectivityView particle_cell_connectivity(m_particle_family);
    command << RUNCOMMAND_ENUMERATE (ParticleLocalId, particle_lid, m_particle_family->allItems())
    {
      Int32 new_cell_lid = (particle_cell_connectivity.cellId(particle_lid) + 1) % max_cell_lid;
      particle_cell_connectivity.setCellId(particle_lid,CellLocalId(new_cell_lid));
      out_particle_cell_lids[particle_lid] = new_cell_lid;
      out_var_particle_cell_lids[particle_lid] = new_cell_lid;
    };
  }

  ENUMERATE_ (Particle, ipart, m_particle_family->allItems()) {
    Particle p = *ipart;
    Int32 cell1_lid = p.cellId();
    Int32 cell2_lid = particle_cell_lids[p.localId()];
    Int32 cell3_lid = var_particle_cell_lids[p];
    if (cell1_lid != cell2_lid)
      ARCANE_FATAL("Incoherent cell for particle p={0} cell1_lid={1} cell2_lid={2}", p.localId(), cell1_lid, cell2_lid);
    if (cell1_lid != cell3_lid)
      ARCANE_FATAL("Incoherent cell for particle p={0} cell1_lid={1} cell3_lid={2}", p.localId(), cell1_lid, cell3_lid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
