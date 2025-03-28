// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUniqueIdMng.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de couche fantômes d'un maillage.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshUniqueIdMng.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IMeshUniqueIdMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshUniqueIdMng::
MeshUniqueIdMng(ITraceMng* tm)
: TraceAccessor(tm)
{
  _initFaceVersion();
  _initEdgeVersion();
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

  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_FACE_UNIQUE_ID_BUILDER_VERSION",true)){
    m_face_builder_version = v.value();
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

void MeshUniqueIdMng::
_initEdgeVersion()
{
  m_edge_builder_version = 1;

  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_EDGE_UNIQUE_ID_BUILDER_VERSION",true)){
    m_edge_builder_version = v.value();
    return;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUniqueIdMng::
setUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId(bool v)
{
  m_use_node_uid_to_generate_edge_and_face_uid = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
