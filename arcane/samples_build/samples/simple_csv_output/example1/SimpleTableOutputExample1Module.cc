// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputExample1Module.cc                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 1 de module utilisant ISimpleTableOutput en tant que singleton.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleTableOutputExample1Module.h"

#include <iostream>
#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [SimpleTableOutputExample1_init]
void SimpleTableOutputExample1Module::
initModule()
{
  // On utilise des valeurs (pseudo-)aléatoires.
  srand(1234);

  // Initialisation du service.
  // On récupère un pointeur vers le singleton créé par Arcane.
  ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();

  // On initialise le tableau grâce à un des initialisateurs.
  // Le nom du tableau sera "Results" et le nom du fichier sortant sera 
  // "Results_Example1.X".
  //
  // On enregistrera le résultat dans le dossier "example1".
  // Au final, on aura un fichier ayant comme chemin :
  // ./output/csv/example1/Results_Example1.X
  //
  // X étant selon le format choisi (.csv par exemple).
  table->init("Results_Example1", "example1");

  // On print le tableau dans son état actuel (vide, avec un titre).
  table->print();
}
//! [SimpleTableOutputExample1_init]

//! [SimpleTableOutputExample1_loop]
void SimpleTableOutputExample1Module::
loopModule()
{
  // On récupère un pointeur vers le singleton créé par Arcane.
  // (on pourrait aussi créer un attribut pour éviter de le récupérer
  // à chaque fois).
  ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();

  // On crée une colonne nommé "Iteration X" (avec X = itération actuelle).
  table->addColumn("Iteration " + String::fromNumber(m_global_iteration()));

  // On génère deux valeurs (c'est pour l'exemple, sinon oui, ça sert à rien).
  Integer nb_fissions = rand()%99;
  Integer nb_collisions = rand()%99;

  // On ajoute deux valeurs à deux lignes (par défaut, 
  // si les lignes n'existe pas encore, elles sont créées).
  table->addElementInRow("Nb de Fissions", nb_fissions);
  table->addElementInRow("Nb de Collisions", nb_collisions);

  // On print le tableau dans son état actuel.
  table->print();

  // On effectue trois itérations.
  if (m_global_iteration() == 3)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}
//! [SimpleTableOutputExample1_loop]

//! [SimpleTableOutputExample1_exit]
void SimpleTableOutputExample1Module::
endModule()
{
  // On récupère un pointeur vers le singleton créé par Arcane.
  ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();

  // On print le tableau dans son état actuel.
  table->print();
  
  // On demande l'écriture du fichier.
  table->writeFile();
}
//! [SimpleTableOutputExample1_exit]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
