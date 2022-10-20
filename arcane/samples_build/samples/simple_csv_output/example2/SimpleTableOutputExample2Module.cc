// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputExample2Module.cc                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 2 de module utilisant ISimpleTableOutput en tant que singleton.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleTableOutputExample2Module.h"

#include <iostream>
#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [SimpleTableOutputExample2_init]
void SimpleTableOutputExample2Module::
initModule()
{
  // On utilise des valeurs (pseudo-)aléatoires.
  srand(1234);

  // Initialisation du service.
  // On récupère un pointeur vers le singleton créé par Arcane.
  ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();
  
  // On initialise le tableau grâce à un des initialisateurs.
  // Le nom du tableau sera par défaut "Results_Example2_default" ou le nom choisi dans 
  // le .arc et le nom du fichier sortant sera "Results_Example2_default.X" (ou Y.X avec
  // Y le nom choisi dans le .arc).
  // X étant selon le format choisi (.csv par exemple).
  if(options()->getTableName() != "")
    table->init(options()->getTableName());
  else
    table->init("Results_Example2_default");

  // On print le tableau dans son état actuel (vide, avec un titre).
  table->print();
}
//! [SimpleTableOutputExample2_init]

//! [SimpleTableOutputExample2_loop]
void SimpleTableOutputExample2Module::
loopModule()
{
  // On récupère un pointeur vers le singleton créé par Arcane.
  // (on pourrait aussi créer un attribut pour éviter le le récupérer à chaque fois).
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
//! [SimpleTableOutputExample2_loop]

//! [SimpleTableOutputExample2_exit]
void SimpleTableOutputExample2Module::
endModule()
{
  // On récupère un pointeur vers le singleton créé par Arcane.
  ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();

  // On print le tableau dans son état actuel.
  table->print();
  
  // On enregistre le résultat dans le dossier par défaut "example2_default" ou le dossier
  // choisi par l'utilisateur dans le .arc.
  // Si l'utilisateur n'a mis ni tableName, ni tableDir dans le .arc,
  // il n'y aura aucune sortie.
  if(options()->getTableName() != "" || options()->getTableDir() != "") {
    if(options()->getTableDir() != "")
      table->setOutputDirectory(options()->getTableDir());
    else
      table->setOutputDirectory("example2_default");

    table->writeFile();
  }
}
//! [SimpleTableOutputExample2_exit]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
