﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUniqueIdMng.cc                                          (C) 2000-2021 */
/*                                                                           */
/* Gestionnaire de couche fantômes d'un maillage.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshUniqueIdMng.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/IMeshUniqueIdMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshUniqueIdMng::
MeshUniqueIdMng(ITraceMng* tm)
: TraceAccessor(tm)
, m_face_builder_version(1)
, m_edge_builder_version(1)
{
  _initFaceVersion();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUniqueIdMng::
setFaceBuilderVersion(Integer n)
{
  if (n<0)
    ARCANE_THROW(ArgumentException,"Bad value for '{0}'<0",n);
  m_face_builder_version = n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUniqueIdMng::
setEdgeBuilderVersion(Integer n)
{
  if (n<0)
    ARCANE_THROW(ArgumentException,"Bad value for '{0}'<0",n);
  m_edge_builder_version = n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUniqueIdMng::
_initFaceVersion()
{
  m_face_builder_version = 1;

  String face_version_str = platform::getEnvironmentVariable("ARCANE_FACE_UNIQUE_ID_BUILDER_VERSION");
  if (!face_version_str.null()){
    Integer v = 0;
    if (builtInGetValue(v,face_version_str))
      ARCANE_FATAL("Invalid value '{0}' for ARCANE_FACE_UNIQUE_ID_BUILDER_VERSION. Value has to be an Integer",
                   face_version_str);
    m_face_builder_version = v;
    return;
  }

  // Pour des raisons de compatibilité avec l'existant, on positionne les
  // valeurs par défaut en fonction de certaines variables d'environnement.
  // Il faudra supprimer ce comportement à terme (car de plus il s'applique
  // à tous les maillages même ceux créés dynamiquement)

  if (!platform::getEnvironmentVariable("ARCANE_NEW_MESHINIT2").null()){
    m_face_builder_version = 3;
    return;
  }

  if (!platform::getEnvironmentVariable("ARCANE_NO_FACE_RENUMBER").null()){
    m_face_builder_version = 0;
    return;
  }

  if (!platform::getEnvironmentVariable("ARCANE_NEW_MESHINIT").null()){
    m_face_builder_version = 2;
    return;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
