// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-wait-elem.js                                         (C) 2000-2022 */
/*                                                                           */
/* Petit script (sans l'utilisation de l'antique jquery) contenant une       */
/* fonction permettant d'attendre qu'un élément soit créé dans la page.      */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile : 
// HTML_EXTRA_FILES      = script-wait-elem.js

// header.html :
// <script type="text/javascript" src="$relpath^script-wait-elem.js"></script>


// Fonction permettant d'attendre que 'expectedElem()' soit chargé
// avant d'appeler 'functionToCall()'.
// @param expectedElem : Fonction retournant l'element à attendre.
// @param functionToCall;item : Fonction à appeler lorsque l'élément sera chargé.
//                               item : l'élément attendu.
var waitElem = (expectedElem, functionToCall) => {
  let elem = expectedElem();
  // Si la création n'a pas eu lieu.
  if (elem == null) {
    // On crée une fonction s'appelant elle-même
    // toutes les 100ms pour attendre l'élement.
    let loopWaitElem = (compt) => {
      elem = expectedElem();

      // En cas de problème (le js, c'est pas une science exacte).
      if (compt > 100 && elem == null) {
        console.log("Impossible de charger elem");
      }
      // Si l'on n'a pas encore trouvé l'élement, on ajoute un
      // évenement 'global' que sera exécuté dans 100ms et qui relancera
      // cette fonction.
      else if (elem == null) {
        window.setTimeout(loopWaitElem, 100, compt + 1);
      }

      // Si l'élement a été trouvé.
      else {
        functionToCall(elem);
      }
    };
    loopWaitElem(0);
  }

  // Si la création a déjà eu lieu, pas besoin d'attendre.
  else {
    functionToCall(elem);
  }
};
