// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleTestCheckpoint.cc                                     (C) 2000-2006 */
/*                                                                           */
/* Module de test des protections/reprises.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/List.h"
#include "arcane/utils/CString.h"

#include "arcane/MeshVariableInfo.h"
#include "arcane/EntryPoint.h"
#include "arcane/ISubDomain.h"
#include "arcane/AbstractModule.h"
#include "arcane/ItemGroup.h"
#include "arcane/ITimeLoop.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/ModuleFactory.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/IMesh.h"
#include "arcane/CommonVariables.h"
#include "arcane/IVariableMng.h"
#include "arcane/ServiceUtils.h"
#include "arcane/IServiceMng.h"
#include "arcane/ICheckpointWriter.h"
#include "arcane/ICheckpointReader.h"
#include "arcane/TimeLoopEntryPointInfo.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/tests/CheckpointTester_axl.h"
#include "arcane/tests/StdArrayMeshVariables.h"
#include "arcane/tests/StdScalarMeshVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test des protections/reprises dans Arcane.
 */
class ModuleTestCheckpoint
: public ArcaneCheckpointTesterObject
{
 public:

  ModuleTestCheckpoint(const ModuleBuildInfo& cb);
  ~ModuleTestCheckpoint();

 public:

  virtual VersionInfo versionInfo() const { return VersionInfo(0,0,1); }

  void test();
  void testInit();
  void restore();

 private:

  StdScalarMeshVariables<Node> m_nodes;
  StdScalarMeshVariables<Face> m_faces;
  StdScalarMeshVariables<Cell> m_cells;
  StdArrayMeshVariables<Node> m_array_nodes;
  StdArrayMeshVariables<Face> m_array_faces;
  StdArrayMeshVariables<Cell> m_array_cells;
  bool m_backward_done;
  Integer m_nb_iteration;
  Integer m_backward_iteration;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(ModuleTestCheckpoint,TestCheckpoint);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleTestCheckpoint::
ModuleTestCheckpoint(const ModuleBuildInfo& mb)
: ArcaneCheckpointTesterObject(mb)
, m_nodes(this,"TestCheckpointNodes")
, m_faces(this,"TestCheckpointFaces")
, m_cells(this,"TestCheckpointCells")
, m_array_nodes(this,"TestCheckpointNodes")
, m_array_faces(this,"TestCheckpointFaces")
, m_array_cells(this,"TestCheckpointCells")
, m_backward_done(false)
, m_nb_iteration(0)
, m_backward_iteration(0)
{
  addEntryPoint(this,"TestCheckpoint_restore",
                &ModuleTestCheckpoint::restore,
                IEntryPoint::WRestore);
  addEntryPoint(this,"TestCheckpoint_test",
                &ModuleTestCheckpoint::test);
  addEntryPoint(this,"TestCheckpoint_testinit",
                &ModuleTestCheckpoint::testInit,
                IEntryPoint::WInit);

  ITimeLoopMng* tlm = subDomain()->timeLoopMng();
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("TestCheckpoint_testinit"));
    clist.add(TimeLoopEntryPointInfo("TestCheckpoint_test"));
    clist.add(TimeLoopEntryPointInfo("TestCheckpoint_restore"));
    ITimeLoop* time_loop = tlm->createTimeLoop("TestCheckpoint");
    time_loop->setEntryPoints("Global",clist);
    tlm->registerTimeLoop(time_loop);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleTestCheckpoint::
~ModuleTestCheckpoint()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleTestCheckpoint::
restore()
{
  info() << "Test checkpoint restore";
  m_global_deltat.assign(m_global_deltat() * 0.5);
  m_backward_done = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleTestCheckpoint::
testInit()
{
  info() << "Test checkpoint init";
  m_array_nodes.initialize();
  m_array_faces.initialize();
  m_array_cells.initialize();
  m_global_deltat = 0.1;

  Integer nb = options()->nbIteration();
  if (nb<=0) nb = 1;
  m_nb_iteration = static_cast<Integer>(nb);

  nb = options()->backwardIteration();
  if (nb<=0) nb = 1;
  m_backward_iteration = static_cast<Integer>(nb);

  nb = options()->backwardPeriod();
  if (nb<=0) nb = 1;
  Integer backward_period = static_cast<Integer>(nb);
  subDomain()->timeLoopMng()->setBackwardSavePeriod(backward_period);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleTestCheckpoint::
test()
{
  info() << "Test checkpoint " << " N = " << m_nb_iteration;
  Integer current_iteration = m_global_iteration();
  if (current_iteration>m_nb_iteration)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
  else if (current_iteration == m_backward_iteration && !m_backward_done)
    subDomain()->timeLoopMng()->goBackward();
  else
  {
    IMesh* mesh = subDomain()->defaultMesh();
    
    // Positionne les valeurs
    {
      m_nodes.setValues(current_iteration, mesh->ownNodes());
      m_faces.setValues(current_iteration, mesh->ownFaces());
      m_cells.setValues(current_iteration, mesh->ownCells());
      m_array_nodes.setValues(current_iteration, mesh->ownNodes());
      m_array_faces.setValues(current_iteration, mesh->ownFaces());
      m_array_cells.setValues(current_iteration, mesh->ownCells());
    }

    IServiceMng* sm = subDomain()->serviceMng();
    String service_name(options()->serviceName());
    ICheckpointWriter* checkpoint_writer = ServiceFinderT<ICheckpointWriter>::find(sm,service_name);
    if (!checkpoint_writer){
      fatal() << "The specified service for protection/restore (" << service_name << ") is not"
              << " available";
    }

    // Ecrit les variables
    IVariableMng* vm = subDomain()->variableMng();
    vm->writeCheckpoint(checkpoint_writer);
    
    // Reinitialise les valeurs avec n'importe quoi
    {
      m_nodes.setValues(0,mesh->ownNodes());
      m_faces.setValues(0,mesh->ownFaces());
      m_cells.setValues(0,mesh->ownCells());
      m_array_nodes.setValues(0,mesh->ownNodes());
      m_array_faces.setValues(0,mesh->ownFaces());
      m_array_cells.setValues(0,mesh->ownCells());
    }
    
    String reader_name = checkpoint_writer->readerServiceName();
    ICheckpointReader* checkpoint_reader = ServiceFinderT<ICheckpointReader>::find(sm,reader_name);
    if (!checkpoint_reader){
      fatal() << "The specified servive to read protection/restore (" << reader_name << ") is not"
              << " available";
    }

    // Lit les variables
    vm->readCheckpoint(checkpoint_reader);
    
    // Vérifie les valeurs
    {
      Integer nb_error = 0;
      nb_error += m_nodes.checkValues(current_iteration,mesh->allNodes());
      nb_error += m_faces.checkValues(current_iteration,mesh->allFaces());
      nb_error += m_cells.checkValues(current_iteration,mesh->allCells());
      nb_error += m_array_nodes.checkValues(current_iteration,mesh->allNodes());
      nb_error += m_array_faces.checkValues(current_iteration,mesh->allFaces());
      nb_error += m_array_cells.checkValues(current_iteration,mesh->allCells());

      if (nb_error!=0)
        fatal() << "Errors in checkValues(): " << nb_error;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
