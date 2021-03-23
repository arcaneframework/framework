// -*- tab-width: 2, indent-tabs-mode: nil, coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LimaMalipp1.cc                                              (C) 2000-2019 */
/*                                                                           */
/* Lecture d'un fichier au format Lima MLI.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <Lima/lima++.h>
#include <Lima/malipp.h>

#include "arcane/cea/LimaMalippT.h"

namespace Arcane
{

extern "C++" IMeshReader::eReturnType
_directLimaPartitionMalipp(ITimerMng* timer_mng,IPrimaryMesh* mesh,
                           const String& filename, Real length_multiplier)
{
  LimaMalippReader<Lima::MaliPPReader> reader(mesh->traceMng());
  IMeshReader::eReturnType rt = reader.readMeshFromFile(timer_mng,mesh, filename, length_multiplier);
  return rt;
}

}
