// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-edit-config-theme.js                                           (C) 2000-2022 */
/*                                                                           */
/* Petit script (sans l'utilisation de l'antique jquery) contenant une       */
/* fonction dédiée à la page "doc_config.md" permettant de modifier les      */
/* cookies dédiés à la personnalisation du thème.                            */
/*                                                                           */
/* Nécessite le script script-helper.js.                                     */
/* Nécessite le script script-config-theme.js.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile : 
// HTML_EXTRA_FILES      = script-helper.js \
//                         script-config-theme.js \
//                         script-edit-config-theme.js

// <script type="text/javascript" src="$relpath^script-helper.js"></script>
// <script type="text/javascript" src="$relpath^script-config-theme.js"></script>
// <script type="text/javascript" src="$relpath^script-edit-config-theme.js"></script>


// Fonction permettant de rendre interactif la page "doc_config.md"
// et de modifier les cookies.
var updateConfigWithCookies = async () => {

  // On recherche les deux items de la partie.
  let span_expand_current_item = document.getElementById("span_expand_current_item");
  let button_apply_expand_current_item = document.getElementById("button_apply_expand_current_item");
  let button_test_expand_current_item = document.getElementById("button_test_expand_current_item");

  // On récupère le cookie de la partie.
  let cookie_expand_current_item = getStorage("expand-current-item");

  // On initialise les deux items selon le cookie.
  switch (cookie_expand_current_item) {
    case "true":
      span_expand_current_item.innerHTML = "Option activée";
      button_apply_expand_current_item.innerHTML = "Désactiver";
      break;
  
    default:
      span_expand_current_item.innerHTML = "Option désactivée";
      button_apply_expand_current_item.innerHTML = "Activer";
      break;
  }

  // On définit une fonction dans le cas où l'utilisateur
  // clique sur le bouton de la partie.
  button_apply_expand_current_item.onclick = () => {
    if (button_apply_expand_current_item.innerHTML == "Activer"){
      span_expand_current_item.innerHTML = "Option activée";
      button_apply_expand_current_item.innerHTML = "Désactiver";
      setStorage("expand-current-item", "true");
    }
    else{
      span_expand_current_item.innerHTML = "Option désactivée";
      button_apply_expand_current_item.innerHTML = "Activer";
      setStorage("expand-current-item", "false");
    }
  };
  button_test_expand_current_item.onclick = () => {
    waitItemExpandCurrent();
  };




  let span_toc_above_all = document.getElementById("span_toc_above_all");
  let button_apply_toc_above_all = document.getElementById("button_apply_toc_above_all");
  let button_test_toc_above_all = document.getElementById("button_test_toc_above_all");

  let cookie_toc_above_all = getStorage("toc-above-all");

  switch (cookie_toc_above_all) {
    case "true":
      span_toc_above_all.innerHTML = "Option activée";
      button_apply_toc_above_all.innerHTML = "Désactiver";
      break;

    default:
      span_toc_above_all.innerHTML = "Option désactivée";
      button_apply_toc_above_all.innerHTML = "Activer";
      break;
  }
  button_apply_toc_above_all.onclick = () => {
    if (button_apply_toc_above_all.innerHTML == "Activer") {
      span_toc_above_all.innerHTML = "Option activée";
      button_apply_toc_above_all.innerHTML = "Désactiver";
      setStorage("toc-above-all", "true");
    }
    else {
      span_toc_above_all.innerHTML = "Option désactivée";
      button_apply_toc_above_all.innerHTML = "Activer";
      setStorage("toc-above-all", "false");
    }
  };
  button_test_toc_above_all.onclick = () => {
    waitItemTocAboveAll();
  };


  let span_apply_old_toc = document.getElementById("span_apply_old_toc");
  let button_apply_apply_old_toc = document.getElementById("button_apply_apply_old_toc");
  let button_test_apply_old_toc = document.getElementById("button_test_apply_old_toc");

  let cookie_apply_old_toc = getStorage("apply-old-toc");

  switch (cookie_apply_old_toc) {
    case "true":
      span_apply_old_toc.innerHTML = "Option activée";
      button_apply_apply_old_toc.innerHTML = "Désactiver";
      break;

    default:
      span_apply_old_toc.innerHTML = "Option désactivée";
      button_apply_apply_old_toc.innerHTML = "Activer";
      break;
  }
  button_apply_apply_old_toc.onclick = () => {
    if (button_apply_apply_old_toc.innerHTML == "Activer") {
      span_apply_old_toc.innerHTML = "Option activée";
      button_apply_apply_old_toc.innerHTML = "Désactiver";
      setStorage("apply-old-toc", "true");
    }
    else {
      span_apply_old_toc.innerHTML = "Option désactivée";
      button_apply_apply_old_toc.innerHTML = "Activer";
      setStorage("apply-old-toc", "false");
    }
  };
  button_test_apply_old_toc.onclick = () => {
    waitItemApplyOldToc();
  };




  let span_expand_level_two = document.getElementById("span_expand_level_two");
  let button_apply_expand_level_two = document.getElementById("button_apply_expand_level_two");
  let button_test_expand_level_two = document.getElementById("button_test_expand_level_two");

  let cookie_expand_level_two = getStorage("expand-level-two");

  switch (cookie_expand_level_two) {
    case "true":
      span_expand_level_two.innerHTML = "Option activée";
      button_apply_expand_level_two.innerHTML = "Désactiver";
      break;

    default:
      span_expand_level_two.innerHTML = "Option désactivée";
      button_apply_expand_level_two.innerHTML = "Activer";
      break;
  }
  button_apply_expand_level_two.onclick = () => {
    if (button_apply_expand_level_two.innerHTML == "Activer") {
      span_expand_level_two.innerHTML = "Option activée";
      button_apply_expand_level_two.innerHTML = "Désactiver";
      setStorage("expand-level-two", "true");
    }
    else {
      span_expand_level_two.innerHTML = "Option désactivée";
      button_apply_expand_level_two.innerHTML = "Activer";
      setStorage("expand-level-two", "false");
    }
  };
  button_test_expand_level_two.onclick = () => {
    waitItemExpandLevelTwo();
  };



  // On recherche les six items de la partie.
  let span_edit_max_width = document.getElementById("span_edit_max_width");
  let range_edit_max_width = document.getElementById("range_edit_max_width");
  let button_max_edit_max_width = document.getElementById("button_max_edit_max_width");
  let button_test_edit_max_width = document.getElementById("button_test_edit_max_width");
  let button_apply_edit_max_width = document.getElementById("button_apply_edit_max_width");
  let button_default_edit_max_width = document.getElementById("button_default_edit_max_width");

  // On récupère le cookie de la partie.
  let cookie_edit_max_width = getStorage("edit-max-width");

  // On récupère la largeur d'origine.
  let default_width = await waitItemGetOriginalContentMaxwidth();

  // Si le cookie n'a pas encore été crée, on le crée.
  if (cookie_edit_max_width == "") {
    cookie_edit_max_width = default_width;
    setStorage("edit-max-width", cookie_edit_max_width);
  }

  // On change la valeur du 'span'.
  span_edit_max_width.innerHTML = cookie_edit_max_width;

  // Si le cookie n'est pas "= 100%" (donc une valeur en "px"),
  // on peut régler le "input range".
  if (cookie_edit_max_width != "100%") {
    range_edit_max_width.value = parseInt(cookie_edit_max_width);
  }

  // Bouton "Max" permettant de définir une largeur 100% de la page.
  button_max_edit_max_width.onclick = () => {
    cookie_edit_max_width = "100%";
    span_edit_max_width.innerHTML = cookie_edit_max_width;
  };

  // Input "range" permettant de définir une largeur personnalisée.
  range_edit_max_width.oninput = () => {
    let value = range_edit_max_width.value;
    cookie_edit_max_width = value + "px";
    span_edit_max_width.innerHTML = value + "px";
  };

  // Bouton permettant de tester la largeur définie.
  button_test_edit_max_width.onclick = () => {
    customMaxWidth = cookie_edit_max_width;
    waitItemChangeMaxWidth();
  };

  // Bouton permettant de changer le cookie.
  button_apply_edit_max_width.onclick = () => {
    setStorage("edit-max-width", cookie_edit_max_width);
  };

  // Bouton permettant de redéfinir la valeur par défaut.
  button_default_edit_max_width.onclick = () => {
    cookie_edit_max_width = default_width;
    customMaxWidth = cookie_edit_max_width;
    waitItemChangeMaxWidth();
    setStorage("edit-max-width", cookie_edit_max_width);
    span_edit_max_width.innerHTML = cookie_edit_max_width;
  };
}