// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-num-lines-code.js                                    (C) 2000-2023 */
/*                                                                           */
/* Petit script (sans l'utilisation de l'antique jquery) permettant          */
/* d'ajouter la numérotation des lignes dans tous les extraits de code.      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile : 
// HTML_EXTRA_STYLESHEET = doxygen-awesome.css
// HTML_EXTRA_FILES      = script-num-lines-code.js

// header.html :
// <script type="text/javascript" src="$relpath^script-num-lines-code.js"></script>


var addLinesNumbers = () => {
  // Liste de tous les class.fragment.
  let allFrag = document.getElementsByClassName("fragment");

  // Si le fragment contient déjà des numéros de lignes.
  if (allFrag.length == 0 || allFrag[0].firstChild.querySelector(".lineno") != null) return;

  for (let fragment of allFrag) {
    let allLine = fragment.getElementsByClassName("line");

    for (let linei = 0; linei < allLine.length; linei++) {
      let first_elem = allLine[linei].firstChild;

      let elem_a = document.createElement("a");
      elem_a.id = linei + 1;

      allLine[linei].insertBefore(elem_a, first_elem);

      let elem_span = document.createElement("span");
      elem_span.className = "lineno";
      elem_span.id = "snippetLineno";
      elem_span.innerHTML = linei + 1;

      allLine[linei].insertBefore(elem_span, first_elem);
    }
  }
};


window.addEventListener('load', function () {
  addLinesNumbers();
});
