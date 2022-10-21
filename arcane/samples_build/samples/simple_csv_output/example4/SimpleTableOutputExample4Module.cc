// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputExample4Module.cc                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 4 de module utilisant ISimpleTableOutput en tant que service.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleTableOutputExample4Module.h"

#include <iostream>
#include <random>

#define NB_ITER 3

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [SimpleTableOutputExample4_init]
void SimpleTableOutputExample4Module::
initModule()
{
  srand(1234);
  options()->stOutput()->init();

  // Pour cet exemple, on va définir le nom des colonnes dès l'init.
  // En effet, ajouter des colonnes au fur et à mesure prend du temps
  // puisque le tableau est réalloué à chaque fois (dans l'implem actuelle).
  // (après, si les perfs de cette partie ne sont pas une priorité, c'est 
  // pas impératif de faire ça, tout dépend de l'utilisation faite du service).
  StringUniqueArray columns_name(NB_ITER);
  for(Integer i = 0; i < NB_ITER; i++){
    columns_name[i] = "Iteration " + String::fromNumber(i+1);
  }
  options()->stOutput()->addColumns(columns_name);

  // On ajoute aussi les deux lignes.
  // Toujours dans un soucis d'optimisation, on peut créer les lignes et récupérer
  // leur position pour la suite, ainsi, on évite deux recherches de String dans un
  // tableau de String à chaque itération.
  options()->stOutput()->addRows(StringUniqueArray{"Nb de Fissions", "Nb de Collisions"});

  m_pos_fis = options()->stOutput()->rowPosition("Nb de Fissions");
  m_pos_col = options()->stOutput()->rowPosition("Nb de Collisions");

  options()->stOutput()->print();
}
//! [SimpleTableOutputExample4_init]

//! [SimpleTableOutputExample4_loop]
void SimpleTableOutputExample4Module::
loopModule()
{
  Integer nb_fissions = rand()%99;
  Integer nb_collisions = rand()%99;

  // On ajoute deux valeurs à nos deux lignes.
  options()->stOutput()->addElementInRow(m_pos_fis, nb_fissions);
  options()->stOutput()->addElementInRow(m_pos_col, nb_collisions);

  options()->stOutput()->print();

  if (m_global_iteration() == NB_ITER)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}
//! [SimpleTableOutputExample4_loop]

//! [SimpleTableOutputExample4_exit]
void SimpleTableOutputExample4Module::
endModule()
{
  // On peut faire la somme des valeurs des lignes si on souhaite.
  // Dans le cas où il y a des cases vides, elle sont initialisé à 0.
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
//! [SimpleTableOutputExample4_exit]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
