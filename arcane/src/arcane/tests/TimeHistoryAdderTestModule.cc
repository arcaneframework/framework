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

#include <arcane/core/ITimeHistoryMng.h>
#include <arcane/core/MeshTimeHistoryAdder.h>
#include <arcane/core/GlobalTimeHistoryAdder.h>

#include "arcane/core/IMesh.h"
#include "arcane/core/internal/ITimeHistoryMngInternal.h"

#include "arcane/tests/TimeHistoryAdderTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
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

void TimeHistoryAdderTestModule::
_writer()
{
  ISubDomain* sd = subDomain();
  Integer iteration = globalIteration();

  sd->timeHistoryMng()->adder()->addValue(TimeHistoryAddValueArg("AAA"), iteration++);
  sd->timeHistoryMng()->adder()->addValue(TimeHistoryAddValueArg("AAA", true, 0), iteration++); // Pas de ++ : normal.
  sd->timeHistoryMng()->adder()->addValue(TimeHistoryAddValueArg("AAA", true, 1), iteration++);

  GlobalTimeHistoryAdder global_adder(sd->timeHistoryMng());
  global_adder.addValue(TimeHistoryAddValueArg("BBB"), iteration++);
  global_adder.addValue(TimeHistoryAddValueArg("BBB", true, 0), iteration++);
  global_adder.addValue(TimeHistoryAddValueArg("BBB", true, 1), iteration++);

  for(auto mesh : sd->meshes()){
    MeshTimeHistoryAdder mesh_adder(sd->timeHistoryMng(), mesh->handle());
    mesh_adder.addValue(TimeHistoryAddValueArg("BBB"), iteration++);
    mesh_adder.addValue(TimeHistoryAddValueArg("BBB", true, 0), iteration++);
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

  sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("AAA")), iterations, values);
  for(Integer i = 0; i < iteration; ++i){
    debug() << "iteration[" << i << "] = " << iterations[i]
            << " == i+1 = " << i+1
            << " -- values[" << i << "] = " << values[i]
            << " == i+(adder++) = " << i+(adder)
    ;
    ARCANE_ASSERT((iterations[i] == i+1), ("Error iterations"));
    ARCANE_ASSERT((values[i] == i+adder), ("Error values"));
  }
  adder++;

  iterations.clear();
  values.clear();
  sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("AAA", true, 0)), iterations, values);
  if(pm->commRank() == 0) {
    for (Integer i = 0; i < iteration; ++i) {
      debug() << "iteration[" << i << "] = " << iterations[i]
              << " == i+1 = " << i+1
              << " -- values[" << i << "] = " << values[i]
              << " == i+(adder++) = " << i+(adder)
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
  sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("AAA", true, 1)), iterations, values);
  if(pm->commRank() == 1) {
    for(Integer i = 0; i < iteration; ++i){
      debug() << "iteration[" << i << "] = " << iterations[i]
              << " == i+1 = " << i+1
              << " -- values[" << i << "] = " << values[i]
              << " == i+(adder++) = " << i+(adder)
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
  sd->timeHistoryMng()->_internalApi()->iterationsAndValues(TimeHistoryAddValueArgInternal(TimeHistoryAddValueArg("BBB")), iterations, values);
  for(Integer i = 0; i < iteration; ++i){
    debug() << "iteration[" << i << "] = " << iterations[i]
            << " == i+1 = " << i+1
            << " -- values[" << i << "] = " << values[i]
            << " == i+(adder++) = " << i+(adder)
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
              << " == i+(adder++) = " << i+(adder)
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
              << " == i+(adder++) = " << i+(adder)
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
              << " == i+(adder++) = " << i+(adder)
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
                << " == i+(adder++) = " << i+(adder)
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
                << " == i+(adder++) = " << i+(adder)
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
