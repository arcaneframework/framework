// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryAdderTestModule.cc                               (C) 2000-2024 */
/*                                                                           */
/* Module de test pour les implementations de ITimeHistoryAdder.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/MeshTimeHistoryAdder.h"
#include "arcane/core/GlobalTimeHistoryAdder.h"

#include "arcane/core/IMesh.h"
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

  VersionInfo versionInfo() const override { return {1, 0, 0}; }
  void init() override;
  void loop() override;
  void _writer();
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

  for(auto mesh : sd->meshes()){

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
  Integer iteration = globalIteration();
  UniqueArray<Int32> iterations;
  UniqueArray<Real> values;

  Integer adder = 1;

  iterations.clear();
  values.clear();
  sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB")), iterations, values);
  for(Integer i = 0; i < iteration; ++i){
    debug() << "iteration[" << i << "] = " << iterations[i]
            << " == i+1 = " << i+1
            << " -- values[" << i << "] = " << values[i]
            << " == i+(adder++) = " << i+adder
    ;
    ARCANE_ASSERT((iterations[i] == i+1), ("Error iterations"));
    ARCANE_ASSERT((values[i] == i+adder), ("Error values"));
  }
  adder++;

  iterations.clear();
  values.clear();
  sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB", true, 0)), iterations, values);
  if(pm->commRank() == 0) {
    for (Integer i = 0; i < iteration; ++i) {
      debug() << "iteration[" << i << "] = " << iterations[i]
              << " == i+1 = " << i+1
              << " -- values[" << i << "] = " << values[i]
              << " == i+(adder++) = " << i+adder
      ;
      ARCANE_ASSERT((iterations[i] == i+1), ("Error iterations"));
      ARCANE_ASSERT((values[i] == i+adder), ("Error values"));
    }
  }
  else{
    ARCANE_ASSERT((iterations.empty()), ("Iterations not empty"));
    ARCANE_ASSERT((values.empty()), ("Values not empty"));
  }
  adder++;

  iterations.clear();
  values.clear();
  sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB", true, 1)), iterations, values);
  if(pm->commRank() == 1) {
    for(Integer i = 0; i < iteration; ++i){
      debug() << "iteration[" << i << "] = " << iterations[i]
              << " == i+1 = " << i+1
              << " -- values[" << i << "] = " << values[i]
              << " == i+(adder++) = " << i+adder
      ;
      ARCANE_ASSERT((iterations[i] == i+1), ("Error iterations"));
      ARCANE_ASSERT((values[i] == i+adder), ("Error values"));
    }
  }
  else{
    ARCANE_ASSERT((iterations.empty()), ("Iterations not empty"));
    ARCANE_ASSERT((values.empty()), ("Values not empty"));
  }
  adder++;

  for(auto mesh : sd->meshes()){
    iterations.clear();
    values.clear();
    sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB"), mesh->handle()), iterations, values);
    for(Integer i = 0; i < iteration; ++i){
      debug() << "iteration[" << i << "] = " << iterations[i]
              << " == i+1 = " << i+1
              << " -- values[" << i << "] = " << values[i]
              << " == i+(adder++) = " << i+adder
      ;
      ARCANE_ASSERT((iterations[i] == i+1), ("Error iterations"));
      ARCANE_ASSERT((values[i] == i+adder), ("Error values"));
    }
    adder++;

    iterations.clear();
    values.clear();
    sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB", true, 0), mesh->handle()), iterations, values);
    if(pm->commRank() == 0) {
      for (Integer i = 0; i < iteration; ++i) {
        debug() << "iteration[" << i << "] = " << iterations[i]
                << " == i+1 = " << i+1
                << " -- values[" << i << "] = " << values[i]
                << " == i+(adder++) = " << i+adder
        ;
        ARCANE_ASSERT((iterations[i] == i+1), ("Error iterations"));
        ARCANE_ASSERT((values[i] == i+adder), ("Error values"));
      }
    }
    else{
      ARCANE_ASSERT((iterations.empty()), ("Iterations not empty"));
      ARCANE_ASSERT((values.empty()), ("Values not empty"));
    }
    adder++;

    iterations.clear();
    values.clear();
    sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB", true, 1), mesh->handle()), iterations, values);
    if(pm->commRank() == 1) {
      for(Integer i = 0; i < iteration; ++i){
        debug() << "iteration[" << i << "] = " << iterations[i]
                << " == i+1 = " << i+1
                << " -- values[" << i << "] = " << values[i]
                << " == i+(adder++) = " << i+adder
        ;
        ARCANE_ASSERT((iterations[i] == i+1), ("Error iterations"));
        ARCANE_ASSERT((values[i] == i+adder), ("Error values"));
      }
    }
    else{
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

ARCANE_REGISTER_MODULE_TIMEHISTORYADDERTEST(TimeHistoryAdderTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
