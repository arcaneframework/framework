// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-resize.js                                            (C) 2000-2022 */
/*                                                                           */
/* Petit script (sans l'utilisation de l'antique jquery) permettant de       */ 
/* réintegrer le redimensionnement du volet de navigation.                   */
/*                                                                           */
/* À utiliser avec 'doxygen-awesome-sidebar-only' uniquement.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile : 
// HTML_EXTRA_STYLESHEET = doxygen-awesome.css \
//                         doxygen-awesome-sidebar-only.css
// HTML_EXTRA_FILES      = script-resize.js

// header.html :
// <script type="text/javascript" src="$relpath^script-resize.js"></script>
// <script type="text/javascript">
//   waitElemForResizeSideNav();
//   setOldSize();
// </script>


let width_border_side_nav = 2;
let min_width_side_nav = 10;
let max_width_side_nav = 1000;
let side_nav = document.getElementsByClassName("ui-resizable-handle ui-resizable-e")[0];

// Fonction permettant de définir les nouvelles propriétés de la side nav.
var ResizeSideNav = () => {
  // On augmente la largeur de la bordure.
  side_nav.style.setProperty('width', width_border_side_nav+"px", "important");

  // On défini la hauteur à 100% de la page.
  side_nav.style.setProperty('height', '100vh');

  // On remonte la bordure.
  side_nav.style.setProperty('top', 'calc(0px - var(--top-height))');

  // On change le curseur.
  side_nav.style.setProperty('cursor', 'col-resize');


  // Évenement appelé lorsque le clique est relaché.
  let stopMoveSideNav = () => {
    // On supprime le listener.
    window.removeEventListener("mousemove", moveSideNav, false);
  };

  // Évenement appelé lorsque l'on clique sur la bordure.
  let mouseDownOnSideNav = (event) => {
    // Pour bloquer la séléction du texte.
    event.preventDefault();

    // On ajoute les évenements sur l'élement 'window' pour que ces évenements
    // puisse agir même si l'on est plus sur la bordure (si on bouge vite la souris par exemple).
    window.addEventListener("mousemove", moveSideNav, false);
    window.addEventListener('mouseup', stopMoveSideNav, false)
  };

  // On calcul la nouvelle position de la bordure.
  // Avec width min et max.
  let moveSideNav = (event) => {
    event.preventDefault();
    let new_pos = () => {
      let n_pos = parseInt(event.pageX, 10) + (width_border_side_nav/2);
      if(n_pos < min_width_side_nav) return min_width_side_nav;
      else if(n_pos > max_width_side_nav) return max_width_side_nav;
      else return n_pos;
    };

    // On modifie la variable CSS '--side-nav-fixed-width'.
    // (vu que doxygen-awesome l'a défini correctement, autant l'utiliser).
    document.querySelector(':root').style.setProperty('--side-nav-fixed-width', new_pos()+'px');
  };

  // On ajoute notre fonction sur l'évenement 'clique down' (dès qu'on clique, sans attendre le relachement).
  side_nav.addEventListener("mousedown", mouseDownOnSideNav, false);
};

// Pour définir les évenements, on doit attendre que la side nav soit
// créée.
let compt = 0;
let waitElemForResizeSideNav = () => {
  // Si la création n'a pas eu lieu.
  if (side_nav == null) {
    // On crée une fonction s'appelant elle-même
    // toutes les 100ms pour attendre l'élement.
    let waitElem = () => {
      compt++;
      side_nav = document.getElementsByClassName("ui-resizable-handle ui-resizable-e")[0];

      // En cas de problème (le js, c'est pas une science exacte).
      if(compt > 100 && side_nav == null){
        console.log("Impossible de charger side_nav");
      }
      // Si l'on n'a pas encore trouvé l'élement, on ajoute un
      // évenement 'global' que sera exécuté dans 100ms et qui relancera
      // cette fonction.
      else if (side_nav == null) {
        window.setTimeout(waitElem, 100);
      }

      // Si l'élement a été trouvé.
      else{
        ResizeSideNav();
      }
    };
    waitElem();
  } 

  // Si la création a déjà eu lieu, pas besoin d'attendre.
  else {
    ResizeSideNav();
  }
};

// Fonction permettant de redéfinir l'ancienne taille
// de side bar.
var setOldSize = () => {

  // Fonction permettant de récupérer un cookie.
  // (source : w3school)
  let getCookie = (cname) => {
    let name = cname + "=";
    let decodedCookie = decodeURIComponent(document.cookie);
    let ca = decodedCookie.split(';');
    for(let i = 0; i <ca.length; i++) {
      let c = ca[i];
      while (c.charAt(0) == ' ') {
        c = c.substring(1);
      }
      if (c.indexOf(name) == 0) {
        return c.substring(name.length, c.length);
      }
    }
    return "";
  }

  // On récupère le cookie gentiment créé et mis à jour par doxygen.
  let coo = getCookie("doxygen_width");
  if(coo != ""){
    document.querySelector(':root').style.setProperty('--side-nav-fixed-width', coo+'px');
  }
};
