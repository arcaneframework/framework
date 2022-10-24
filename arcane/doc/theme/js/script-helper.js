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


// Fonction permettant d'attendre que 'expectedItem()' soit chargé
// avant d'appeler 'functionToCall()'.
// @param expectedItem : Fonction retournant l'element à attendre.
// @param functionToCall;item : Fonction à appeler lorsque l'élément sera chargé.
//                               item : l'élément attendu.
// @return Une promesse.
var waitItem = (expectedItem, functionToCall) => {
  return new Promise((resolve, reject) => {

    let item = expectedItem();

    // Si la création n'a pas eu lieu.
    if (item == null) {
      // On crée une fonction s'appelant elle-même
      // toutes les 100ms pour attendre l'élement.
      let loopWaitItem = (compt) => {
        item = expectedItem();

        // En cas de problème (le js, c'est pas une science exacte).
        if (compt > 100 && item == null) {
          reject(() => {console.log("Impossible de charger item");});
        }
        // Si l'on n'a pas encore trouvé l'élement, on ajoute un
        // évenement 'global' que sera exécuté dans 100ms et qui relancera
        // cette fonction.
        else if (item == null) {
          window.setTimeout(loopWaitItem, 20, compt + 1);
        }

        // Si l'élement a été trouvé.
        else {
          resolve(functionToCall(item));
        }
      };
      loopWaitItem(0);
    }

    // Si la création a déjà eu lieu, pas besoin d'attendre.
    else {
      resolve(functionToCall(item));
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

var removeCookie = (cname) => {
  setCookie(cname, "", -1);
}


// Fonction permettant de récupérer une valeur du stockage
// selon le navigateur.
var getStorage = (cname) => {
  if (window.chrome)
  return localStorage.getItem(cname);
  else
  return getCookie(cname);
}

// Fonction permettant de stocker une valeur
// selon le navigateur.
var setStorage = (cname, cvalue, exdays = 400) => {
  if (window.chrome)
    localStorage.setItem(cname, cvalue);
  else
    setCookie(cname, cvalue, exdays);
}

// Fonction permettant de supprimer une valeur du stockage
// selon le navigateur.
var removeStorage = (cname) => {
  if (window.chrome)
    localStorage.removeItem(cname);
  else
    removeCookie(cname);
}
