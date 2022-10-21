// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-cookies.js                                           (C) 2000-2022 */
/*                                                                           */
/* Petit script (sans l'utilisation de l'antique jquery) contenant une       */
/*       */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile : 
// HTML_EXTRA_FILES      = script-cookies.js

// <script type="text/javascript" src="$relpath^script-cookies.js"></script>


var updateConfigWithCookies = async () => {
  let span_expand_current = document.getElementById("span_expand_current");
  let button_expand_current = document.getElementById("button_expand_current");
  let cookie_expand_current = getCookie("expand-current");

  switch (cookie_expand_current) {
    case "true":
      span_expand_current.innerHTML = "Option activée";
      button_expand_current.innerHTML = "Désactiver";
      break;
  
    default:
      span_expand_current.innerHTML = "Option désactivée";
      button_expand_current.innerHTML = "Activer";
      break;
  }

  button_expand_current.onclick = () => {
    if (button_expand_current.innerHTML == "Activer"){
      span_expand_current.innerHTML = "Option activée";
      button_expand_current.innerHTML = "Désactiver";
    }
    else{
      span_expand_current.innerHTML = "Option désactivée";
      button_expand_current.innerHTML = "Activer";
    }
  };




  let span_change_toc_pos = document.getElementById("span_change_toc_pos");
  let button_change_toc_pos = document.getElementById("button_change_toc_pos");
  let cookie_change_toc_pos = getCookie("toc_pos");

  switch (cookie_change_toc_pos) {
    case "true":
      span_change_toc_pos.innerHTML = "Option activée";
      button_change_toc_pos.innerHTML = "Désactiver";
      break;

    default:
      span_change_toc_pos.innerHTML = "Option désactivée";
      button_change_toc_pos.innerHTML = "Activer";
      break;
  }
  button_change_toc_pos.onclick = () => {
    if (button_change_toc_pos.innerHTML == "Activer") {
      span_change_toc_pos.innerHTML = "Option activée";
      button_change_toc_pos.innerHTML = "Désactiver";
    }
    else {
      span_change_toc_pos.innerHTML = "Option désactivée";
      button_change_toc_pos.innerHTML = "Activer";
    }
  };



  let span_change_old_toc = document.getElementById("span_change_old_toc");
  let button_change_old_toc = document.getElementById("button_change_old_toc");
  let cookie_change_old_toc = getCookie("old_toc");

  switch (cookie_change_old_toc) {
    case "true":
      span_change_old_toc.innerHTML = "Option activée";
      button_change_old_toc.innerHTML = "Désactiver";
      break;

    default:
      span_change_old_toc.innerHTML = "Option désactivée";
      button_change_old_toc.innerHTML = "Activer";
      break;
  }
  button_change_old_toc.onclick = () => {
    if (button_change_old_toc.innerHTML == "Activer") {
      span_change_old_toc.innerHTML = "Option activée";
      button_change_old_toc.innerHTML = "Désactiver";
    }
    else {
      span_change_old_toc.innerHTML = "Option désactivée";
      button_change_old_toc.innerHTML = "Activer";
    }
  };



  let span_change_max_width = document.getElementById("span_change_max_width");
  let range_change_max_width = document.getElementById("range_change_max_width");
  let button_change_max_width = document.getElementById("button_max_max_width");
  let buttonTest_change_max_width = document.getElementById("button_test_max_width");
  let buttonApply_change_max_width = document.getElementById("button_apply_max_width");
  let buttonDefault_change_max_width = document.getElementById("button_default_max_width");
  let cookie_change_max_width = getCookie("max_width");

  let default_width = await waitItemGetOriginalContentMaxwidth();

  if (cookie_change_max_width == ""){
    cookie_change_max_width = default_width;
    setCookie("max_width", cookie_change_max_width);
  }
  
  span_change_max_width.innerHTML = cookie_change_max_width;
  if (cookie_change_max_width != "100%"){
    range_change_max_width.value = parseInt(cookie_change_max_width);
  }

  button_change_max_width.onclick = () => {
    cookie_change_max_width = "100%";
    span_change_max_width.innerHTML = cookie_change_max_width;
  };

  range_change_max_width.oninput = () => {
    let value = range_change_max_width.value;
    cookie_change_max_width = value + "px";
    span_change_max_width.innerHTML = value+"px";
  };

  buttonTest_change_max_width.onclick = () => {
    customMaxWidth = cookie_change_max_width;
    waitItemChangeMaxWidth();
  };

  buttonApply_change_max_width.onclick = () => {
    setCookie("max_width", cookie_change_max_width);
  };

  buttonDefault_change_max_width.onclick = () => {
    cookie_change_max_width = default_width;
    customMaxWidth = cookie_change_max_width;
    waitItemChangeMaxWidth();
    setCookie("max_width", cookie_change_max_width);
    span_change_max_width.innerHTML = cookie_change_max_width;
  };






  let span_expand_level = document.getElementById("span_expand_level");
  let button_expand_level = document.getElementById("button_expand_level");
  let cookie_expand_level = getCookie("expand_level");

  switch (cookie_expand_level) {
    case "true":
      span_expand_level.innerHTML = "Option activée";
      button_expand_level.innerHTML = "Désactiver";
      break;

    default:
      span_expand_level.innerHTML = "Option désactivée";
      button_expand_level.innerHTML = "Activer";
      break;
  }
  button_expand_level.onclick = () => {
    if (button_expand_level.innerHTML == "Activer") {
      span_expand_level.innerHTML = "Option activée";
      button_expand_level.innerHTML = "Désactiver";
    }
    else {
      span_expand_level.innerHTML = "Option désactivée";
      button_expand_level.innerHTML = "Activer";
    }
  };


}