// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableComparatorExample1Module.cc                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 1 de module utilisant ISimpleTableOutput et ISimpleTableComparator*/
/* en tant que singleton.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleTableComparatorExample1Module.h"
#include <arcane/impl/ArcaneMain.h>

#include <iostream>
#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [SimpleTableComparatorExample1_init]
void SimpleTableComparatorExample1Module::
initModule()
{
  srand(1234);
  ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();
  table->init("Results_Example1", "example1");
}
//! [SimpleTableComparatorExample1_init]

//! [SimpleTableComparatorExample1_loop]
void SimpleTableComparatorExample1Module::
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
//! [SimpleTableComparatorExample1_loop]

//! [SimpleTableComparatorExample1_exit]
void SimpleTableComparatorExample1Module::
endModule()
{
  ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();
  ISimpleTableComparator* comparator = ServiceBuilder<ISimpleTableComparator>(subDomain()).getSingleton();

  comparator->init(table);

  if(!comparator->isReferenceExist()){
    info() << "Écriture du fichier de référence";
    comparator->writeReferenceFile();
  }

  else {
    if(comparator->compareWithReference()){
      info() << "Mêmes valeurs !!!";
    }

    else{
      error() << "Valeurs différentes :(";
      IArcaneMain::arcaneMain()->setErrorCode(1);
    }
  }
  
  table->writeFile();
}
//! [SimpleTableComparatorExample1_exit]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
