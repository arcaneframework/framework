// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEventsImpl.cc                                           (C) 2000-2023 */
/*                                                                           */
/* Implémentation des évènements sur le maillage.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshEventsImpl.h"

#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EventObservable<const MeshEventArgs&>& MeshEventsImpl::
eventObservable(eMeshEventType type)
{
  switch (type) {
  case eMeshEventType::BeginPrepareDump:
    return m_on_begin_prepare_dump;
  case eMeshEventType::EndPrepareDump:
    return m_on_end_prepare_dump;
  }
  ARCANE_FATAL("Unknown event '{0}'", (int)type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
