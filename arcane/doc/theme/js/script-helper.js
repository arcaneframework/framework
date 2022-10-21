// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-helper.js                                            (C) 2000-2022 */
/*                                                                           */
/* Petit script (sans l'utilisation de l'antique jquery) contenant des       */
/* fonctions génériques pouvant être uilisées par les autres scripts.        */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile : 
// HTML_EXTRA_FILES      = script-helper.js

// header.html :
// <script type="text/javascript" src="$relpath^script-helper.js"></script>


// Fonction permettant d'attendre que 'expectedElem()' soit chargé
// avant d'appeler 'functionToCall()'.
// @param expectedElem : Fonction retournant l'element à attendre.
// @param functionToCall;item : Fonction à appeler lorsque l'élément sera chargé.
//                               item : l'élément attendu.
var waitItem = (expectedElem, functionToCall) => {
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

var waitItemPromise = (expectedElem, functionToCall) => {
  return new Promise((resolve, reject) => {

    let elem = expectedElem();

    // Si la création n'a pas eu lieu.
    if (elem == null) {
      // On crée une fonction s'appelant elle-même
      // toutes les 100ms pour attendre l'élement.
      let loopWaitElem = (compt) => {
        elem = expectedElem();

        // En cas de problème (le js, c'est pas une science exacte).
        if (compt > 100 && elem == null) {
          reject("Impossible de charger elem");
        }
        // Si l'on n'a pas encore trouvé l'élement, on ajoute un
        // évenement 'global' que sera exécuté dans 100ms et qui relancera
        // cette fonction.
        else if (elem == null) {
          window.setTimeout(loopWaitElem, 100, compt + 1);
        }

        // Si l'élement a été trouvé.
        else {
          resolve(functionToCall(elem));
        }
      };
      loopWaitElem(0);
    }

    // Si la création a déjà eu lieu, pas besoin d'attendre.
    else {
      resolve(functionToCall(elem));
    }

  });

}



// Fonction permettant de récupérer un cookie.
// (source : w3school)
var getCookie = (cname) => {
  let name = cname + "=";
  let decodedCookie = decodeURIComponent(document.cookie);
  let ca = decodedCookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length).replace("#per#", "%");
    }
  }
  return "";
}

// Fonction permettant d'écrire un cookie.
// (source : w3school)
var setCookie = (cname, cvalue, exdays=400) => {
  cvalue = cvalue.replace("%", "#per#");
  const d = new Date();
  d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
  let expires = "expires=" + d.toUTCString();
  document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}
