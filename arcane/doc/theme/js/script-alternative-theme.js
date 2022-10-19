// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-alternative-theme.js                                 (C) 2000-2022 */
/*                                                                           */
/*           */
/*       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Encore experimental.
// Appel manuel via la console JS :
// F12 -> Console -> ChangeMaxWidth()
// F12 -> Console -> ChangeTocPos()
// F12 -> Console -> ChangeOldToc()
// F12 -> Console -> ExpendLevel(2)
// F12 -> Console -> ExpendCurrent()

var originalValue = null;
var stepMaxWidth = false;
var stepTocPos = false;
var stepOldToc = false;
var stepExpend = false;

var ChangeMaxWidth = () => {

  if (originalValue == null){
    originalValue = document.querySelector(":root").style.getPropertyValue("--content-maxwidth");
  }

  if (stepMaxWidth){
    document.querySelector(":root").style.setProperty("--content-maxwidth", originalValue);
    stepMaxWidth = false;
  }
  else {
    document.querySelector(":root").style.setProperty("--content-maxwidth", "100%");
    stepMaxWidth = true;
  }
};

var ChangeTocPos = () => {

  if (stepTocPos) {
    document.getElementsByClassName("toc interactive")[0].style.setProperty("position", "sticky");
    document.getElementsByClassName("toc interactive")[0].style.setProperty("z-index", "initial");
    stepTocPos = false;
  }
  else {
    document.getElementsByClassName("toc interactive")[0].style.setProperty("position", "absolute");
    document.getElementsByClassName("toc interactive")[0].style.setProperty("z-index", "1");
    stepTocPos = true;
  }
};

var ChangeOldToc = () => {

  if (stepOldToc) {
    document.getElementsByClassName("contents")[0].style.setProperty("display", "flex");
    document.getElementsByClassName("toc interactive")[0].style.setProperty("position", "sticky");
    stepOldToc = false;
  }
  else {
    document.getElementsByClassName("contents")[0].style.setProperty("display", "inherit");
    document.getElementsByClassName("toc interactive")[0].style.setProperty("position", "inherit");
    stepOldToc = true;
  }
};

var nodeSaved = null;

var ExpandLevel = (level) => {
  let symbol = "►";
  if(stepExpend){
    symbol = "▼";
    stepExpend = false;
  }
  else{
    stepExpend = true;
  }

  let GetAllElemLevel = (level) => {
    let str = "";
    for(let i = 0; i < level; i++){
      str += "ul > li > ";
    }
    return document.querySelectorAll("#nav-tree-contents > " + str + "div > a");
  };

  GetAllElemLevel(level).forEach(
    (node, _) => {
      if (node.querySelector("span").innerHTML == symbol){
        node.onclick();
      }
      else if (stepExpend){
        nodeSaved = node;
      }
    }
  );
  if (!stepExpend){
    nodeSaved.onclick();
  }
};

var ExpandCurrent = () => {
  document.getElementsByClassName("item selected")[0].querySelector("a").onclick();
};
