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

void SimpleTableOutputExample4Module::
initModule()
{
  // On utilise des valeurs (pseudo-)aléatoires.
  srand(1234);

  // Initialisation du service.

  // On initialise le tableau grâce à un des initialisateurs.
  // Le nom du tableau sera le nom choisi dans le .arc.
  options()->csvOutput()->init();

  // Pour cet exemple, on va définir le nom des colonnes dès l'init.
  // En effet, ajouter des colonnes au fur et à mesure prend du temps
  // puisque le tableau est réalloué à chaque fois (dans l'implem actuelle).
  // (après, si les perfs de cette partie ne sont pas une priorité, c'est 
  // pas impératif de faire ça, tout dépend de l'utilisation faite du service).
  StringUniqueArray columns_name(NB_ITER);
  for(Integer i = 0; i < NB_ITER; i++){
    columns_name[i] = "Iteration " + String::fromNumber(i+1);
  }
  options()->csvOutput()->addColumns(columns_name);

  // On ajoute aussi les deux lignes.
  // Toujours dans un soucis d'optimisation, on peut créer les lignes et récupérer
  // leur position pour la suite, ainsi, on évite deux recherches de String dans un
  // tableau de String à chaque itération.
  options()->csvOutput()->addRows(StringUniqueArray{"Nb de Fissions", "Nb de Collisions"});

  pos_fis = options()->csvOutput()->rowPosition("Nb de Fissions");
  pos_col = options()->csvOutput()->rowPosition("Nb de Collisions");

  // On print le tableau dans son état actuel (vide, avec un titre).
  options()->csvOutput()->print();
}

void SimpleTableOutputExample4Module::
loopModule()
{
  // On génère deux valeurs (c'est pour l'exemple, sinon oui, ça sert à rien).
  Integer nb_fissions = rand()%99;
  Integer nb_collisions = rand()%99;

  // On ajoute deux valeurs à nos deux lignes.
  options()->csvOutput()->addElementInRow(pos_fis, nb_fissions);
  options()->csvOutput()->addElementInRow(pos_col, nb_collisions);

  // On print le tableau dans son état actuel.
  options()->csvOutput()->print();

  // On effectue trois itérations.
  if (m_global_iteration() == NB_ITER)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

void SimpleTableOutputExample4Module::
endModule()
{
  // On peut faire la somme des valeurs des lignes si on souhaite.
  // Dans le cas où il y a des cases vides, elle sont initialisé à 0 
  // (TODO : mais pas lors d'une redim, alors que c'est ce qu'on voudrai).
  for(Integer pos = 0; pos < options()->csvOutput()->numberOfRows(); pos++) {
    RealUniqueArray row = options()->csvOutput()->row(pos);
    Real sum = 0.;
    for(Real elem : row) {
      sum += elem;
    }
    options()->csvOutput()->addElementInColumn("Somme", sum);
  }

  // On print le tableau dans son état actuel.
  options()->csvOutput()->print();
  
  // On enregistre le résultat dans le dossier choisi
  // par l'utilisateur dans le .arc.
  options()->csvOutput()->writeFile();
  
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
