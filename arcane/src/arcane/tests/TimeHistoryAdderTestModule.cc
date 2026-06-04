// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryAdderTestModule.cc                               (C) 2000-2025 */
/*                                                                           */
/* Test module for the implementations of ITimeHistoryAdder.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/GlobalTimeHistoryAdder.h"
#include "arcane/core/MeshTimeHistoryAdder.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/internal/ITimeHistoryMngInternal.h"

#include "arcane/tests/TimeHistoryAdderTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TimeHistoryAdderTestModule
: public ArcaneTimeHistoryAdderTestObject
{
 public:

  explicit TimeHistoryAdderTestModule(const ModuleBuildInfo& mbi);

 public:

  VersionInfo versionInfo() const override { return { 1, 0, 0 }; }
  void init() override;
  void loop() override;
  void _writer();
  void _writer_example1();
  void _writer_example2();
  void _checker();
  void exit() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeHistoryAdderTestModule::
TimeHistoryAdderTestModule(const ModuleBuildInfo& mbi)
: ArcaneTimeHistoryAdderTestObject(mbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryAdderTestModule::
init()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryAdderTestModule::
loop()
{
  _writer();
  _writer_example1();
  _writer_example2();
  _checker();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryAdderTestModule::
_writer()
{
  ISubDomain* sd = subDomain();
  Integer iteration = globalIteration();

  // Creation of a global adder.
  GlobalTimeHistoryAdder global_adder(sd->timeHistoryMng());

  // We add a value to the "BBB" history common to all subdomains.
  global_adder.addValue(TimeHistoryAddValueArg("BBB"), iteration++);

  // We add a value to the "BBB" history specific to subdomain 0.
  global_adder.addValue(TimeHistoryAddValueArg("BBB", true, 0), iteration++);

  // We add a value to the "BBB" history specific to subdomain 1.
  global_adder.addValue(TimeHistoryAddValueArg("BBB", true, 1), iteration++);

  for (auto mesh : sd->meshes()) {

    // Creation of a mesh-specific adder.
    MeshTimeHistoryAdder mesh_adder(sd->timeHistoryMng(), mesh->handle());

    // We add a value to the "BBB" history specific to mesh and common to all subdomains.
    mesh_adder.addValue(TimeHistoryAddValueArg("BBB"), iteration++);

    // We add a value to the "BBB" history specific to mesh and subdomain 0.
    mesh_adder.addValue(TimeHistoryAddValueArg("BBB", true, 0), iteration++);

    // We add a value to the "BBB" history specific to mesh and subdomain 1.
    mesh_adder.addValue(TimeHistoryAddValueArg("BBB", true, 1), iteration++);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryAdderTestModule::
_checker()
{
  ISubDomain* sd = subDomain();
  IParallelMng* pm = parallelMng();
  IParallelReplication* pr = pm->replication();
  Integer iteration = globalIteration();
  UniqueArray<Int32> iterations;
  UniqueArray<Real> values;

  bool master_io = pr->hasReplication() ? (pr->isMasterRank() && pm->isMasterIO()) : pm->isMasterIO();

  Integer adder = 1;

  iterations.clear();
  values.clear();
  sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB")), iterations, values);
  if (master_io) {
    for (Integer i = 0; i < iteration; ++i) {
      debug() << "iteration[" << i << "] = " << iterations[i]
              << " == i+1 = " << i + 1
              << " -- values[" << i << "] = " << values[i]
              << " == i+(adder++) = " << i + adder;
      ARCANE_ASSERT((iterations[i] == i + 1), ("Error iterations"));
      ARCANE_ASSERT((values[i] == i + adder), ("Error values"));
    }
  }
  else {
    ARCANE_ASSERT((iterations.empty()), ("Iterations not empty"));
    ARCANE_ASSERT((values.empty()), ("Values not empty"));
  }
  adder++;

  iterations.clear();
  values.clear();
  sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB", true, 0)), iterations, values);
  if (pm->commRank() == 0 && pr->isMasterRank()) {
    for (Integer i = 0; i < iteration; ++i) {
      debug() << "iteration[" << i << "] = " << iterations[i]
              << " == i+1 = " << i + 1
              << " -- values[" << i << "] = " << values[i]
              << " == i+(adder++) = " << i + adder;
      ARCANE_ASSERT((iterations[i] == i + 1), ("Error iterations"));
      ARCANE_ASSERT((values[i] == i + adder), ("Error values"));
    }
  }
  else {
    ARCANE_ASSERT((iterations.empty()), ("Iterations not empty"));
    ARCANE_ASSERT((values.empty()), ("Values not empty"));
  }
  adder++;

  iterations.clear();
  values.clear();
  sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB", true, 1)), iterations, values);
  if (pm->commRank() == 1 && pr->isMasterRank()) {
    for (Integer i = 0; i < iteration; ++i) {
      debug() << "iteration[" << i << "] = " << iterations[i]
              << " == i+1 = " << i + 1
              << " -- values[" << i << "] = " << values[i]
              << " == i+(adder++) = " << i + adder;
      ARCANE_ASSERT((iterations[i] == i + 1), ("Error iterations"));
      ARCANE_ASSERT((values[i] == i + adder), ("Error values"));
    }
  }
  else {
    ARCANE_ASSERT((iterations.empty()), ("Iterations not empty"));
    ARCANE_ASSERT((values.empty()), ("Values not empty"));
  }
  adder++;

  for (auto mesh : sd->meshes()) {
    iterations.clear();
    values.clear();
    sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB"), mesh->handle()), iterations, values);
    if (master_io) {
      for (Integer i = 0; i < iteration; ++i) {
        debug() << "iteration[" << i << "] = " << iterations[i]
                << " == i+1 = " << i + 1
                << " -- values[" << i << "] = " << values[i]
                << " == i+(adder++) = " << i + adder;
        ARCANE_ASSERT((iterations[i] == i + 1), ("Error iterations"));
        ARCANE_ASSERT((values[i] == i + adder), ("Error values"));
      }
    }
    else {
      ARCANE_ASSERT((iterations.empty()), ("Iterations not empty"));
      ARCANE_ASSERT((values.empty()), ("Values not empty"));
    }
    adder++;

    iterations.clear();
    values.clear();
    sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB", true, 0), mesh->handle()), iterations, values);
    if (pm->commRank() == 0 && pr->isMasterRank()) {
      for (Integer i = 0; i < iteration; ++i) {
        debug() << "iteration[" << i << "] = " << iterations[i]
                << " == i+1 = " << i + 1
                << " -- values[" << i << "] = " << values[i]
                << " == i+(adder++) = " << i + adder;
        ARCANE_ASSERT((iterations[i] == i + 1), ("Error iterations"));
        ARCANE_ASSERT((values[i] == i + adder), ("Error values"));
      }
    }
    else {
      ARCANE_ASSERT((iterations.empty()), ("Iterations not empty"));
      ARCANE_ASSERT((values.empty()), ("Values not empty"));
    }
    adder++;

    iterations.clear();
    values.clear();
    sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB", true, 1), mesh->handle()), iterations, values);
    if (pm->commRank() == 1 && pr->isMasterRank()) {
      for (Integer i = 0; i < iteration; ++i) {
        debug() << "iteration[" << i << "] = " << iterations[i]
                << " == i+1 = " << i + 1
                << " -- values[" << i << "] = " << values[i]
                << " == i+(adder++) = " << i + adder;
        ARCANE_ASSERT((iterations[i] == i + 1), ("Error iterations"));
        ARCANE_ASSERT((values[i] == i + adder), ("Error values"));
      }
    }
    else {
      ARCANE_ASSERT((iterations.empty()), ("Iterations not empty"));
      ARCANE_ASSERT((values.empty()), ("Values not empty"));
    }
    adder++;
  }
  iterations.clear();
  values.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryAdderTestModule::
exit()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryAdderTestModule::
_writer_example1()
{
  //![snippet_timehistory_example1]
  // Calculation of the subdomain pressure average.
  Real avg_pressure_subdomain = 0;
  ENUMERATE_ (Cell, icell, mesh()->ownCells()) {
    avg_pressure_subdomain += m_pressure[icell];
  }
  if (!mesh()->ownCells().empty()) {
    avg_pressure_subdomain /= mesh()->ownCells().size();
  }

  ISubDomain* sd = subDomain();
  IParallelMng* pm = parallelMng();
  Integer my_rank = pm->commRank();
  Integer nb_proc = pm->commSize();

  // Calculation of the global average pressure.
  Real avg_pressure_global = pm->reduce(IParallelMng::eReduceType::ReduceSum, avg_pressure_subdomain);
  avg_pressure_global /= nb_proc;

  // Creation of the object allowing values to be added to a value history.
  GlobalTimeHistoryAdder global_adder(sd->timeHistoryMng());

  // Adding the avg_pressure_subdomain value to the "avg_pressure" history. One history per subdomain.
  global_adder.addValue(TimeHistoryAddValueArg("avg_pressure", true, my_rank), avg_pressure_subdomain);

  // Adding the avg_pressure_global value to the "avg_pressure" history. Global history.
  global_adder.addValue(TimeHistoryAddValueArg("avg_pressure"), avg_pressure_global);
  //![snippet_timehistory_example1]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryAdderTestModule::
_writer_example2()
{
  //![snippet_timehistory_example2]
  ISubDomain* sd = subDomain();
  IParallelMng* pm = parallelMng();
  Integer my_rank = pm->commRank();
  Integer nb_proc = pm->commSize();
  Integer nb_mesh = sd->meshes().size();

  // Will contain the subdomain pressure average ("(2)" in the image).
  Real avg_pressure_subdomain = 0;

  // Will contain the pressure average of the entire domain ("(4)" in the image)
  Real avg_pressure_global = 0;

  // For each mesh.
  for (auto mesh : sd->meshes()) {

    // Will contain the subdomain and mesh "mesh" pressure average ("(1)" in the image).
    Real avg_pressure_subdomain_mesh = 0;
    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
      avg_pressure_subdomain_mesh += m_pressure[icell];
    }
    if (!mesh->ownCells().empty()) {
      avg_pressure_subdomain_mesh /= mesh->ownCells().size();
    }

    // Will contain the pressure average of the entire domain and mesh "mesh" ("(3)" in the image).
    Real avg_pressure_global_mesh = pm->reduce(IParallelMng::eReduceType::ReduceSum, avg_pressure_subdomain_mesh);
    avg_pressure_global_mesh /= nb_proc;

    // Creation of the object allowing values to be added to a value history linked to mesh "mesh".
    MeshTimeHistoryAdder mesh_adder(sd->timeHistoryMng(), mesh->handle());

    // Adding the avg_pressure_subdomain_mesh value to the "avg_pressure" history linked to mesh "mesh".
    // One history per subdomain.
    mesh_adder.addValue(TimeHistoryAddValueArg("avg_pressure", true, my_rank), avg_pressure_subdomain_mesh);

    // Adding the avg_pressure_global value to the "avg_pressure" history linked to mesh "mesh".
    // Global history.
    mesh_adder.addValue(TimeHistoryAddValueArg("avg_pressure"), avg_pressure_global_mesh);

    avg_pressure_subdomain += avg_pressure_subdomain_mesh;
    avg_pressure_global += avg_pressure_global_mesh;
  }

  if (nb_mesh != 0) {
    avg_pressure_subdomain /= nb_mesh;
    avg_pressure_global /= nb_mesh;
  }

  // Creation of the object allowing values to be added to a value history.
  GlobalTimeHistoryAdder global_adder(sd->timeHistoryMng());

  // Adding the avg_pressure_subdomain value to the "avg_pressure" history. One history per subdomain.
  global_adder.addValue(TimeHistoryAddValueArg("avg_pressure", true, my_rank), avg_pressure_subdomain);

  // Adding the avg_pressure_global value to the "avg_pressure" history. Global history.
  global_adder.addValue(TimeHistoryAddValueArg("avg_pressure"), avg_pressure_global);
  //![snippet_timehistory_example2]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_TIMEHISTORYADDERTEST(TimeHistoryAdderTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
