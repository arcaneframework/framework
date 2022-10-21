// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableComparatorExample3Module.cc                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 3 de module utilisant ISimpleTableOutput en tant que singleton.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleTableComparatorExample3Module.h"
#include <arcane/impl/ArcaneMain.h>

#include <iostream>
#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [SimpleTableComparatorExample3_init]
void SimpleTableComparatorExample3Module::
initModule()
{
  srand(1234);
  ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();
  table->init("Results_Example3", "example3");
}
//! [SimpleTableComparatorExample3_init]

//! [SimpleTableComparatorExample3_loop]
void SimpleTableComparatorExample3Module::
loopModule()
{
  ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();

  table->addColumn("Iteration " + String::fromNumber(m_global_iteration()));

  Integer nb_fissions = rand()%99;
  Integer nb_collisions = rand()%99;

  table->addElementInRow("Nb de Fissions", nb_fissions);
  table->addElementInRow("Nb de Collisions", nb_collisions);

  if (m_global_iteration() == 3)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}
//! [SimpleTableComparatorExample3_loop]

//! [SimpleTableComparatorExample3_exit]
void SimpleTableComparatorExample3Module::
endModule()
{
  ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();

  options()->stComparator()->init(table);

  if(!options()->stComparator()->isReferenceExist()){
    info() << "Écriture du fichier de référence";
    //table->editElement("Iteration 1", "Nb de Fissions", 9999);
    table->editElement("Iteration 1", "Nb de Collisions", 9999);
    options()->stComparator()->writeReferenceFile();
  }

  else {
    options()->stComparator()->editRegexRows("^.*Fissions.*$");

    if(options()->stComparator()->compareWithReference()){
      info() << "Mêmes valeurs !!!";
    }

    else{
      error() << "Valeurs différentes :(";
      IArcaneMain::arcaneMain()->setErrorCode(1);
    }
  }

  table->writeFile();
}
//! [SimpleTableComparatorExample3_exit]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
