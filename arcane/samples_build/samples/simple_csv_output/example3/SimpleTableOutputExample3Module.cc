// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputExample3Module.cc                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 3 de module utilisant ISimpleTableOutput en tant que service.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleTableOutputExample3Module.h"

#include <iostream>
#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [SimpleTableOutputExample3_init]
void SimpleTableOutputExample3Module::
initModule()
{
  srand(1234);

  // On initialise le tableau grâce à un des initialisateurs.
  // Le nom du tableau sera le nom choisi dans le .arc.
  options()->stOutput()->init();

  // On print le tableau dans son état actuel (vide, avec un titre).
  options()->stOutput()->print();
}
//! [SimpleTableOutputExample3_init]

//! [SimpleTableOutputExample3_loop]
void SimpleTableOutputExample3Module::
loopModule()
{
  // On crée une colonne nommé "Iteration X" (avec X = itération actuelle).
  options()->stOutput()->addColumn("Iteration " + String::fromNumber(m_global_iteration()));

  // On génère deux valeurs (c'est pour l'exemple, sinon oui, ça sert à rien).
  Integer nb_fissions = rand()%99;
  Integer nb_collisions = rand()%99;

  // On ajoute deux valeurs à deux lignes (par défaut, 
  // si les lignes n'existe pas encore, elles sont créées).
  options()->stOutput()->addElementInRow("Nb de Fissions", nb_fissions);
  options()->stOutput()->addElementInRow("Nb de Collisions", nb_collisions);

  // On print le tableau dans son état actuel.
  options()->stOutput()->print();

  // On effectue trois itérations.
  if (m_global_iteration() == 3)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}
//! [SimpleTableOutputExample3_loop]

//! [SimpleTableOutputExample3_exit]
void SimpleTableOutputExample3Module::
endModule()
{
  // On print le tableau dans son état actuel.
  options()->stOutput()->print();
  
  // On enregistre le résultat dans le dossier choisi
  // par l'utilisateur dans le .arc.
  options()->stOutput()->writeFile();
}
//! [SimpleTableOutputExample3_exit]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
