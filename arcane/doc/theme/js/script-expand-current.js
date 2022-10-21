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
/* Nécessite le script script-wait-elem.js.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile : 
// HTML_EXTRA_FILES      = script-wait-elem.js \
//                         script-expand-current.js

// header.html :
// <script type="text/javascript" src="$relpath^script-wait-elem.js"></script>
// <script type="text/javascript" src="$relpath^script-expand-current.js"></script>
// <script type="text/javascript">
//   waitCurrentItem();
// </script>


// On attend et on "clique" sur la flèche pour étendre le menu.
let waitCurrentItem = () => {
  waitElem(
    () => { return document.getElementsByClassName("item selected")[0]; },
    (item) => { item.querySelector("a").onclick(); }
  );
};

