// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableComparatorExample2Module.cc                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 2 de module utilisant ISimpleTableOutput et ISimpleTableComparator*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleTableComparatorExample2Module.h"
#include <arcane/impl/ArcaneMain.h>

#include <iostream>
#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [SimpleTableComparatorExample2_init]
void SimpleTableComparatorExample2Module::
initModule()
{
  srand(1234);
  options()->stOutput()->init();
}
//! [SimpleTableComparatorExample2_init]

//! [SimpleTableComparatorExample2_loop]
void SimpleTableComparatorExample2Module::
loopModule()
{
  options()->stOutput()->addColumn("Iteration " + String::fromNumber(m_global_iteration()));

  Integer nb_fissions = rand()%99;
  Integer nb_collisions = rand()%99;

  options()->stOutput()->addElementInRow("Nb de Fissions", nb_fissions);
  options()->stOutput()->addElementInRow("Nb de Collisions", nb_collisions);

  if (m_global_iteration() == 3)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}
//! [SimpleTableComparatorExample2_loop]

//! [SimpleTableComparatorExample2_exit]
void SimpleTableComparatorExample2Module::
endModule()
{
  options()->stComparator()->init(options()->stOutput());

  if(!options()->stComparator()->isReferenceExist()){
    info() << "Écriture du fichier de référence";
    options()->stComparator()->writeReferenceFile();
  }

  else {
    if(options()->stComparator()->compareWithReference()){
      info() << "Mêmes valeurs !!!";
    }

    else{
      error() << "Valeurs différentes :(";
      IArcaneMain::arcaneMain()->setErrorCode(1);
    }
  }

  options()->stOutput()->writeFile();
}
//! [SimpleTableComparatorExample2_exit]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
