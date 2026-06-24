// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LimaUtils.cc                                                (C) 2000-2026 */
/*                                                                           */
/* Lima reader for 'mli2' format.                                            */
/*---------------------------------------------------------------------------*/

#include "arcane/lima/internal/LimaUtils.h"

#include "arcane/lima/internal/LimaMalippT.h"

#ifdef ARCANE_LIMA_HAS_MLI2
#include <Lima/malipp2.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType LimaUtils::
_directLimaPartitionMalipp2(ITimerMng* timer_mng, IPrimaryMesh* mesh,
                            const String& filename, Real length_multiplier)
{
#ifdef ARCANE_LIMA_HAS_MLI2
  LimaMalippReader<Lima::MaliPPReader2> reader(mesh->traceMng());
  auto rt = reader.readMeshFromFile(timer_mng, mesh, filename, length_multiplier);
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
