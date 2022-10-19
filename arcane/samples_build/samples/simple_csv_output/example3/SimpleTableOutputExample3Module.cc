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

void SimpleTableOutputExample3Module::
initModule()
{
  // On utilise des valeurs (pseudo-)aléatoires.
  srand(1234);

  // Initialisation du service.

  // On initialise le tableau grâce à un des initialisateurs.
  // Le nom du tableau sera le nom choisi dans le .arc.
  options()->csvOutput()->init();

  // On print le tableau dans son état actuel (vide, avec un titre).
  options()->csvOutput()->print();
}

void SimpleTableOutputExample3Module::
loopModule()
{
  // On crée une colonne nommé "Iteration X" (avec X = itération actuelle).
  options()->csvOutput()->addColumn("Iteration " + String::fromNumber(m_global_iteration()));

  // On génère deux valeurs (c'est pour l'exemple, sinon oui, ça sert à rien).
  Integer nb_fissions = rand()%99;
  Integer nb_collisions = rand()%99;

  // On ajoute deux valeurs à deux lignes (par défaut, 
  // si les lignes n'existe pas encore, elles sont créées).
  options()->csvOutput()->addElementInRow("Nb de Fissions", nb_fissions);
  options()->csvOutput()->addElementInRow("Nb de Collisions", nb_collisions);

  // On print le tableau dans son état actuel.
  options()->csvOutput()->print();

  // On effectue trois itérations.
  if (m_global_iteration() == 3)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

void SimpleTableOutputExample3Module::
endModule()
{
  // On print le tableau dans son état actuel.
  options()->csvOutput()->print();
  
  // On enregistre le résultat dans le dossier choisi
  // par l'utilisateur dans le .arc.
  options()->csvOutput()->writeFile();
  
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
