// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputExample5Module.cc                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 5 de module utilisant ISimpleTableOutput en tant que service.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleTableOutputExample5Module.h"

#include <iostream>
#include <random>

#define NB_ITER 3

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [SimpleTableOutputExample5_init]
void SimpleTableOutputExample5Module::
initModule()
{
  srand(1234);
  options()->stOutput()->init();

  StringUniqueArray columns_name(NB_ITER);
  for(Integer i = 0; i < NB_ITER; i++){
    columns_name[i] = "Iteration " + String::fromNumber(i+1);
  }
  options()->stOutput()->addColumns(columns_name);

  // On ajoute aussi quatre lignes.
  // Toujours dans un soucis d'optimisation, on peut créer les lignes et récupérer
  // leur position pour la suite, ainsi, on évite deux recherches de String dans un
  // tableau de String à chaque itération.
  options()->stOutput()->addRows(StringUniqueArray{
    "Nb de Fissions", 
    "Nb de Fissions (div par 2)", 
    "Nb de Collisions", 
    "Nb de Collisions (div par 2)"});

  pos_fis = options()->stOutput()->rowPosition("Nb de Fissions");
  pos_col = options()->stOutput()->rowPosition("Nb de Collisions");

  options()->stOutput()->print();
}
//! [SimpleTableOutputExample5_init]

//! [SimpleTableOutputExample5_loop]
void SimpleTableOutputExample5Module::
loopModule()
{
  Integer nb_fissions = rand()%99;
  Integer nb_collisions = rand()%99;

  // Ici, on utilise editElementDown() pour ajouter un élement sous l'élement dernièrement
  // modifié.
  options()->stOutput()->addElementInRow(pos_fis, nb_fissions);
  options()->stOutput()->editElementDown(nb_fissions/2.);
  options()->stOutput()->addElementInRow(pos_col, nb_collisions);
  options()->stOutput()->editElementDown(nb_collisions/2.);

  options()->stOutput()->print();
  if (m_global_iteration() == NB_ITER)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}
//! [SimpleTableOutputExample5_loop]

//! [SimpleTableOutputExample5_exit]
void SimpleTableOutputExample5Module::
endModule()
{
  for(Integer pos = 0; pos < options()->stOutput()->numberOfRows(); pos++) {
    RealUniqueArray row = options()->stOutput()->row(pos);
    Real sum = 0.;
    for(Real elem : row) {
      sum += elem;
    }
    options()->stOutput()->addElementInColumn("Somme", sum);
  }

  options()->stOutput()->print();
  options()->stOutput()->writeFile();
}
//! [SimpleTableOutputExample5_exit]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
