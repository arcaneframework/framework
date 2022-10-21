// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-expand-current.js                                    (C) 2000-2022 */
/*                                                                           */
/* Petit script (sans l'utilisation de l'antique jquery) permettant de       */
/* montrer les pages d'un chapitre dans la navbar.                           */
/*                                                                           */
/* Nécessite le script script-helper.js.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile : 
// HTML_EXTRA_FILES      = script-helper.js \
//                         script-expand-current.js

// header.html :
// <script type="text/javascript" src="$relpath^script-helper.js"></script>
// <script type="text/javascript" src="$relpath^script-expand-current.js"></script>
// <script type="text/javascript">
//   waitItemExpandCurrent();
// </script>

var expandCurrent = (item) => {
  item.querySelector("a").onclick();
}

// On attend et on "clique" sur la flèche pour étendre le menu.
var waitItemExpandCurrent = () => {
  waitItem(
    () => { return document.getElementsByClassName("item selected")[0]; },
    expandCurrent
  );
};



var changeTocPos = (item) => {
  if (stepTocPos) {
    item.style.setProperty("position", "sticky");
    item.style.setProperty("z-index", "initial");
    stepTocPos = false;
  }
  else {
    item.style.setProperty("position", "absolute");
    item.style.setProperty("z-index", "1");
    stepTocPos = true;
  }
};

var waitItemChangeTocPos = () => {
  waitItem(
    () => { return document.getElementsByClassName("toc interactive")[0]; },
    changeTocPos
  );
};




var changeOldToc = (items) => {

  if (stepOldToc) {
    items.contents.style.setProperty("display", "flex");
    items.toc.style.setProperty("position", "sticky");
    stepOldToc = false;
  }
  else {
    items.contents.style.setProperty("display", "inherit");
    items.toc.style.setProperty("position", "inherit");
    stepOldToc = true;
  }
};

var waitItemChangeOldToc = () => {
  let items = () => {
    let contents = document.getElementsByClassName("contents")[0];
    let toc = document.getElementsByClassName("toc interactive")[0];
    if (contents == null || toc == null) return null;
    return { contents: contents, toc: toc };
  };
  waitItem(items, changeOldToc);
};



var getOriginalContentMaxwidth = (item) => {
  return getComputedStyle(item).getPropertyValue("--content-maxwidth");
}

var waitItemGetOriginalContentMaxwidth = () => {
  return waitItemPromise(
    () => { return document.body; },
    getOriginalContentMaxwidth
  );
};


var customMaxWidth = "1000px";

var changeMaxWidth = (item) => {
  return item.style.setProperty("--content-maxwidth", customMaxWidth);
};

var waitItemChangeMaxWidth = () => {
  waitItem(
    () => { return document.querySelector(":root"); },
    changeMaxWidth
  );
};










var nodeSaved = null;

var expandLevel = (item) => {
  let symbol = "►";
  if (stepExpend) {
    symbol = "▼";
    stepExpend = false;
  }
  else {
    stepExpend = true;
  }
  item.forEach(
    (node, _) => {
      if (node.querySelector("span").innerHTML == symbol) {
        node.onclick();
      }
      else if (stepExpend) {
        nodeSaved = node;
      }
    }
  );
  if (!stepExpend) {
    nodeSaved.onclick();
  }
};

var waitItemExpandLevel = () => {
  waitItem(
    () => { return document.querySelectorAll("#nav-tree-contents > ul > li > ul > li > div > a"); },
    expandLevel
  );
};


var changeFunctionButton = (item) => {
  item.onclick = () => {
    waitItemExpandLevel();
  };
};

var waitItemChangeFunctionButton = () => {
  waitItem(
    () => { return document.getElementById("nav-sync"); },
    changeFunctionButton
  );
};
