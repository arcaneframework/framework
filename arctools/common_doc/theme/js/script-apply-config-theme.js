// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-apply-config-theme.js                                (C) 2000-2023 */
/*                                                                           */
/* Petit script (sans l'utilisation de l'antique jquery) permettant d'       */
/* appliquer la configuration du thème.                                      */
/*                                                                           */
/* La configuration se trouve dans les cookies et est définie par            */
/* "script-edit-config-theme.js" (et par la page "doc_config.md").           */
/*                                                                           */
/* Les fonctions permettant d'appliquer la configuration se trouvent dans    */
/* "script-config-theme.js".                                                 */
/*                                                                           */
/* Nécessite le script script-helper.js.                                     */
/* Nécessite le script script-config-theme.js.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile : 
// HTML_EXTRA_FILES      = script-helper.js \
//                         script-config-theme.js \
//                         script-apply-config-theme.js

// header.html :
// <script type="text/javascript" src="$relpath^script-helper.js"></script>
// <script type="text/javascript" src="$relpath^script-config-theme.js"></script>
// <script type="text/javascript" src="$relpath^script-apply-config-theme.js"></script>
// <script type="text/javascript">
//   applyConfigWithCookies();
// </script>



/**
 * Fonction permettant d'appeler les fonctions correspondantes aux
 * options de configurations demandées.
 */
var applyConfigWithCookies = async () => {
  
  // Si cette variable vaut true, c'est qu'on ne veut pas appliquer la
  // personnalisation du thème.
  if(no_custom_theme) return;


  // Ici, on récupère le cookie et on regarde si on doit appliquer
  // l'option de configuration correspondante.
  let cookie = getStorage("expand-current-item");

  switch (cookie) {
    case "true":
      waitItemExpandCurrent();
      break;

    default:
      break;
  }

  cookie = getStorage("toc-above-all");

  switch (cookie) {
    case "true":
      waitItemTocAboveAll();
      break;

    default:
      break;
  }

  cookie = getStorage("apply-old-toc");

  switch (cookie) {
    case "true":
      waitItemApplyOldToc();
      break;

    default:
      break;
  }

  cookie = getStorage("expand-level-two");

  switch (cookie) {
    case "true":
      waitItemChangeFunctionButton();
      break;

    default:
      break;
  }



  cookie = getStorage("edit-max-width");

  // S'il n'y a pas de cookie définit, on l'initialise.
  if (cookie == "") {
    cookie = await waitItemGetOriginalContentMaxwidth();
    setStorage("edit-max-width", cookie);
  }

  // Cette varialbe est une variable globale définit
  // dans le fichier "script-config-theme.js".
  customMaxWidth = cookie;
  waitItemChangeMaxWidth();
}