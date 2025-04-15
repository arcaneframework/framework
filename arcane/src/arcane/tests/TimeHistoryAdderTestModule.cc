// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryAdderTestModule.cc                               (C) 2000-2025 */
/*                                                                           */
/* Module de test pour les implementations de ITimeHistoryAdder.             */
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

  // Création d'un adder global.
  GlobalTimeHistoryAdder global_adder(sd->timeHistoryMng());

  // On ajoute une valeur à l'historique "BBB" commun à tous les sous-domaines.
  global_adder.addValue(TimeHistoryAddValueArg("BBB"), iteration++);

  // On ajoute une valeur à l'historique "BBB" spécifique au sous-domaine 0.
  global_adder.addValue(TimeHistoryAddValueArg("BBB", true, 0), iteration++);

  // On ajoute une valeur à l'historique "BBB" spécifique au sous-domaine 1.
  global_adder.addValue(TimeHistoryAddValueArg("BBB", true, 1), iteration++);

  for (auto mesh : sd->meshes()) {

    // Création d'un adder spécifique au maillage mesh.
    MeshTimeHistoryAdder mesh_adder(sd->timeHistoryMng(), mesh->handle());

    // On ajoute une valeur à l'historique "BBB" spécifique au maillage mesh et commun à tous les sous-domaines.
    mesh_adder.addValue(TimeHistoryAddValueArg("BBB"), iteration++);

    // On ajoute une valeur à l'historique "BBB" spécifique au maillage mesh et au sous-domaine 0.
    mesh_adder.addValue(TimeHistoryAddValueArg("BBB", true, 0), iteration++);

    // On ajoute une valeur à l'historique "BBB" spécifique au maillage mesh et au sous-domaine 1.
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
  // Calcul de la moyenne des pressions du sous-domaine.
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

  // Calcul de la pression moyenne globale.
  Real avg_pressure_global = pm->reduce(IParallelMng::eReduceType::ReduceSum, avg_pressure_subdomain);
  avg_pressure_global /= nb_proc;

  // Création de l'objet permettant d'ajouter des valeurs dans un historique des valeurs.
  GlobalTimeHistoryAdder global_adder(sd->timeHistoryMng());

  // Ajout de la valeur avg_pressure_subdomain dans l'historique "avg_pressure". Un historique par sous-domaine.
  global_adder.addValue(TimeHistoryAddValueArg("avg_pressure", true, my_rank), avg_pressure_subdomain);

  // Ajout de la valeur avg_pressure_global dans l'historique "avg_pressure". Historique global.
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

  // Contiendra la moyenne des pressions du sous-domaine ("(2)" sur l'image).
  Real avg_pressure_subdomain = 0;

  // Contiendra la moyenne des pressions de tout le domaine ("(4)" sur l'image)
  Real avg_pressure_global = 0;

  // Pour chaque maillage.
  for (auto mesh : sd->meshes()) {

    // Contiendra la moyenne des pressions du sous-domaines et du maillage "mesh" ("(1)" sur l'image).
    Real avg_pressure_subdomain_mesh = 0;
    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
      avg_pressure_subdomain_mesh += m_pressure[icell];
    }
    if (!mesh->ownCells().empty()) {
      avg_pressure_subdomain_mesh /= mesh->ownCells().size();
    }

    // Contiendra la moyenne des pressions de tout le domaine et du maillage "mesh" ("(3)" sur l'image).
    Real avg_pressure_global_mesh = pm->reduce(IParallelMng::eReduceType::ReduceSum, avg_pressure_subdomain_mesh);
    avg_pressure_global_mesh /= nb_proc;

    // Création de l'objet permettant d'ajouter des valeurs dans un historique des valeurs lié au maillage "mesh".
    MeshTimeHistoryAdder mesh_adder(sd->timeHistoryMng(), mesh->handle());

    // Ajout de la valeur avg_pressure_subdomain_mesh dans l'historique "avg_pressure" lié au maillage "mesh".
    // Un historique par sous-domaine.
    mesh_adder.addValue(TimeHistoryAddValueArg("avg_pressure", true, my_rank), avg_pressure_subdomain_mesh);

    // Ajout de la valeur avg_pressure_global dans l'historique "avg_pressure" lié au maillage "mesh".
    // Historique global.
    mesh_adder.addValue(TimeHistoryAddValueArg("avg_pressure"), avg_pressure_global_mesh);

    avg_pressure_subdomain += avg_pressure_subdomain_mesh;
    avg_pressure_global += avg_pressure_global_mesh;
  }

  if (nb_mesh != 0) {
    avg_pressure_subdomain /= nb_mesh;
    avg_pressure_global /= nb_mesh;
  }

  // Création de l'objet permettant d'ajouter des valeurs dans un historique des valeurs.
  GlobalTimeHistoryAdder global_adder(sd->timeHistoryMng());

  // Ajout de la valeur avg_pressure_subdomain dans l'historique "avg_pressure". Un historique par sous-domaine.
  global_adder.addValue(TimeHistoryAddValueArg("avg_pressure", true, my_rank), avg_pressure_subdomain);

  // Ajout de la valeur avg_pressure_global dans l'historique "avg_pressure". Historique global.
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
