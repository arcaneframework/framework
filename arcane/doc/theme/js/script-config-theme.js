// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-config-theme.js                                    (C) 2000-2022 */
/*                                                                           */
/* Petit script (sans l'utilisation de l'antique jquery) contenant les       */
/* fonctions permettant de modifier l'apparence d'une page.                  */
/*                                                                           */
/* Les appels de ces fonctions sont réalisés par                             */
/* "script-apply-config-theme.js" et dans le fichier "doc_config.md".        */
/*                                                                           */
/* Nécessite le script script-helper.js.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile : 
// HTML_EXTRA_FILES      = script-helper.js \
//                         script-config-theme.js

// header.html :
// <script type="text/javascript" src="$relpath^script-helper.js"></script>
// <script type="text/javascript" src="$relpath^script-config-theme.js"></script>


// Fonction permettant d'étendre l'item courant.
var expandCurrent = (item) => {
  // Doxygen étend l'item Arcane automatiquement.
  if (item.innerText != "▼Arcane"){
    item.querySelector("a").onclick();
  }
}
// Fonction permettant d'attendre que l'item courant soit accessible.
var waitItemExpandCurrent = () => {
  waitItem(
    () => { return document.getElementsByClassName("item selected")[0]; },
    expandCurrent
  );
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

var stepTocAboveAll = false;

// Fonction permettant d'afficher le toc au-dessus du texte.
var tocAboveAll = (item) => {
  if (stepTocAboveAll) {
    item.style.setProperty("position", "sticky");
    item.style.setProperty("z-index", "initial");
    stepTocAboveAll = false;
  }
  else {
    item.style.setProperty("position", "absolute");
    item.style.setProperty("z-index", "1");
    stepTocAboveAll = true;
  }
};
var waitItemTocAboveAll = () => {
  waitItem(
    () => { return document.getElementsByClassName("toc interactive")[0]; },
    tocAboveAll
  );
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

var stepApplyOldToc = false;

// Fonction permettant de remplacer le nouveau toc par le toc d'origine.
var applyOldToc = (items) => {

  if (stepApplyOldToc) {
    items.contents.style.setProperty("display", "flex");
    items.toc.style.setProperty("position", "sticky");
    stepApplyOldToc = false;
  }
  else {
    items.contents.style.setProperty("display", "inherit");
    items.toc.style.setProperty("position", "inherit");
    stepApplyOldToc = true;
  }
};
var waitItemApplyOldToc = () => {
  let items = () => {
    let contents = document.getElementsByClassName("contents")[0];
    let toc = document.getElementsByClassName("toc interactive")[0];
    if (contents == null || toc == null) return null;
    return { contents: contents, toc: toc };
  };
  waitItem(items, applyOldToc);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

var stepExpendLevelTwo = false;
var nodeSaved = null;

// Fonction permettant d'étendre le niveau deux de la nav bar.
var expandLevelTwo = (item) => {
  // Pour déterminer quels items étendre, on regarde "l'orientation"
  // de la flèche.
  let symbol = "►";
  if (stepExpendLevelTwo) {
    symbol = "▼";
    stepExpendLevelTwo = false;
  }
  else {
    stepExpendLevelTwo = true;
  }
  // Pour tous les items, on regarde l'orientation de la flèche et
  // selon, on étend/rétracte le niveau.
  item.forEach(
    (node, _) => {
      if (node.querySelector("span").innerHTML == symbol) {
        node.onclick();
      }
      // Si l'utilisateur a étendu un niveau, on le sauvegarde pour
      // le restaurer plus tard.
      else if (stepExpendLevelTwo) {
        nodeSaved = node;
      }
    }
  );
  // Si on est en mode "rétracte", on reétend l'item sauvé.
  if (!stepExpendLevelTwo && nodeSaved != null) {
    nodeSaved.onclick();
    nodeSaved = null;
  }
};
var waitItemExpandLevelTwo = () => {
  waitItem(
    () => { return document.querySelectorAll("#nav-tree-contents > ul > li > ul > li > div > a"); },
    expandLevelTwo
  );
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Fonction permettant d'ajouter un bouton à coté du bouton de
// synchronisation permettant l'extension de niveau (fonction au-dessus).
var changeFunctionButton = (item) => {

  // Image du bouton.
  let image = document.createElement("img");
  image.setAttribute("src", "../../sync_on.png");
  image.setAttribute("title", "Etendre/Rétracter menus");

  // Le div du bouton.
  let divExtend = document.createElement("div");
  divExtend.appendChild(image);
  // L'id pour l'apparence CSS.
  divExtend.setAttribute("id", "nav-extend");
  divExtend.onclick = () => {
    waitItemExpandLevelTwo();
  };

  item.appendChild(divExtend);
};
var waitItemChangeFunctionButton = () => {
  waitItem(
    () => { return document.getElementById("nav-tree-contents"); },
    changeFunctionButton
  );
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Fonction permettant de récupérer la largeur d'origine.
var getOriginalContentMaxwidth = (item) => {
  return getComputedStyle(item).getPropertyValue("--content-maxwidth");
}
var waitItemGetOriginalContentMaxwidth = () => {
  return waitItem(
    () => { return document.body; },
    getOriginalContentMaxwidth
  );
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

var customMaxWidth = "1000px";

// Fonction permettant de définir une autre largeur.
var changeMaxWidth = (item) => {
  return item.style.setProperty("--content-maxwidth", customMaxWidth);
};
var waitItemChangeMaxWidth = () => {
  waitItem(
    () => { return document.querySelector(":root"); },
    changeMaxWidth
  );
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
