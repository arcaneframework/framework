// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputExample6Module.cc                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 6 de module utilisant ISimpleTableOutput en tant que service.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "SimpleTableOutputExample6Module.h"

#include <iostream>
#include <random>

#define NB_ITER 20

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [SimpleTableOutputExample6_init]
void SimpleTableOutputExample6Module::
initModule()
{
  options()->stOutput()->init();

  StringUniqueArray names(NB_ITER);
  for(Integer i = 0; i < NB_ITER; i++){
    names[i] = String::fromNumber(i);
  }
  options()->stOutput()->addColumns(names);
  options()->stOutput()->addRows(names);

  options()->stOutput()->editElement(0, 0, 1);

  options()->stOutput()->print();
}
//! [SimpleTableOutputExample6_init]

//! [SimpleTableOutputExample6_loop]
void SimpleTableOutputExample6Module::
loopModule()
{
  Real elem1 = 0;

  // On récupère la valeur au début de la ligne et on met à jour le pointeur.
  Real elem2 = options()->stOutput()->element(0, m_global_iteration()-1, true);

  for(Integer i = 0; i < m_global_iteration()+1; i++){
    // La case sous le pointeur prend la valeur "elem1+elem2".
    // On demande à ce que le pointeur ne se mette pas à jour.
    options()->stOutput()->editElementDown(elem1 + elem2, false);
    elem1 = elem2;
    // On récupère l'élement à droite du pointeur et on force la mise à jour du pointeur.
    elem2 = options()->stOutput()->elementRight(true);
  }

  if (m_global_iteration() == NB_ITER)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}
//! [SimpleTableOutputExample6_loop]

//! [SimpleTableOutputExample6_exit]
void SimpleTableOutputExample6Module::
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
//! [SimpleTableOutputExample6_exit]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
