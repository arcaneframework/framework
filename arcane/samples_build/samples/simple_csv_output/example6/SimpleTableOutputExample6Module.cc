// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputExample6Module.cc                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 5 de module utilisant ISimpleTableOutput en tant que service.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleTableOutputExample6Module.h"

#include <iostream>
#include <random>

#define NB_ITER 20

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleTableOutputExample6Module::
initModule()
{
  // Initialisation du service.

  // On initialise le tableau grâce à un des initialisateurs.
  // Le nom du tableau sera le nom choisi dans le .arc.
  options()->csvOutput()->init();

  // Pour cet exemple, on va définir le nom des colonnes dès l'init.
  // En effet, ajouter des colonnes au fur et à mesure prend du temps
  // puisque le tableau est réalloué à chaque fois (dans l'implem actuelle).
  // (après, si les perfs de cette partie ne sont pas une priorité, c'est 
  // pas impératif de faire ça, tout dépend de l'utilisation faite du service).
  StringUniqueArray names(NB_ITER);
  for(Integer i = 0; i < NB_ITER; i++){
    names[i] = String::fromNumber(i);
  }
  options()->csvOutput()->addColumns(names);
  options()->csvOutput()->addRows(names);

  options()->csvOutput()->editElement(0, 0, 1);

  // On print le tableau dans son état actuel.
  options()->csvOutput()->print();
}

void SimpleTableOutputExample6Module::
loopModule()
{
  // On peut aussi utiliser les déplacements par direction.
  // Pour ça, dans le service, on a un "pointeur" "last_elem" qui pointe
  // vers la dernière case modifiée. On peut aussi demander à element() 
  // de forcer la mise à jour du pointeur (ce "pointeur", c'est juste deux
  // Integer qui désignent la position (x, y) de la dernière case modifiée).

  // Les cases qui n'existent pas renvoi 0 et ne mettent pas à jour le pointeur
  // (vu qu'il ne peut pas "pointer" vers une case qui n'existe pas).
  Real elem1 = options()->csvOutput()->element(-1, m_global_iteration()-1, true);

  // On récupère la valeur au début de la ligne et on met à jour le pointeur.
  Real elem2 = options()->csvOutput()->element(0, m_global_iteration()-1, true);

  for(Integer i = 0; i < m_global_iteration()+1; i++){
    // La case sous le pointeur prend la valeur "elem1+elem2".
    // On demande à ce que le pointeur ne se mette pas à jour.
    options()->csvOutput()->editElementDown(elem1 + elem2, false);
    elem1 = elem2;
    // On récupère l'élement à droite du pointeur et on force la mise à jour du pointeur.
    elem2 = options()->csvOutput()->elementRight(true);
  }

  // On effectue vingt itérations.
  if (m_global_iteration() == NB_ITER)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

void SimpleTableOutputExample6Module::
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
