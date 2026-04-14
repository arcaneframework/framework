// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* script-doxygen-version.js                                   (C) 2000-2026 */
/*                                                                           */
/* Script permettant de découper le numéro de version de Doxygen en trois    */
/* variables number.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Utilisation :
// doxyfile :
// HTML_EXTRA_FILES      = script-doxygen-version.js

// header.html :
// <script type="text/javascript" src="$relpath^script-doxygen-version.js"></script>
// <script type="text/javascript">
//   setDoxygenVersion("$doxygenversion");
// </script>

// Note : $doxygenversion sera remplacé par Doxygen lors de la génération de
// la documentation.

var doxygen_major_version = 0;
var doxygen_minor_version = 0;
var doxygen_fix_version = 0;

var setDoxygenVersion = (version) => {
  let versionParts = version.split('.');
  doxygen_major_version = parseInt(versionParts[0]);
  doxygen_minor_version = parseInt(versionParts[1]);
  doxygen_fix_version = parseInt(versionParts[2]);
};
