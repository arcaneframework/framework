// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LimaMalipp.cc                                               (C) 2000-2026 */
/*                                                                           */
/* Lima reader for 'mli' format.                                             */
/*---------------------------------------------------------------------------*/

#include "arcane/lima/internal/LimaUtils.h"

#include "arcane/lima/internal/LimaMalippT.h"

#ifdef ARCANE_LIMA_HAS_MLI
#include <Lima/malipp.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType LimaUtils::
_directLimaPartitionMalipp(ITimerMng* timer_mng, IPrimaryMesh* mesh,
                           const String& filename, Real length_multiplier)
{
#ifdef ARCANE_LIMA_HAS_MLI
  LimaMalippReader<Lima::MaliPPReader> reader(mesh->traceMng());
  IMeshReader::eReturnType rt = reader.readMeshFromFile(timer_mng, mesh, filename, length_multiplier);
  return rt;
#else
  ARCANE_UNUSED(timer_mng);
  ARCANE_UNUSED(mesh);
  ARCANE_UNUSED(filename);
  ARCANE_UNUSED(length_multiplier);
  return IMeshReader::eReturnType::RTIrrelevant;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
