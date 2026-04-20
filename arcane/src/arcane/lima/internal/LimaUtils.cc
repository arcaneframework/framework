// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LimaUtils.cc                                                (C) 2000-2026 */
/*                                                                           */
/* Fonctions utilitaires pour Lima.                                          */
/*---------------------------------------------------------------------------*/

#include "arcane/lima/internal/LimaUtils.h"

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/ItemGroup.h"

#include "arcane/lima/internal/LimaMalippT.h"

#include <algorithm>

#ifdef ARCANE_LIMA_HAS_MLI
#include <Lima/malipp.h>
#endif
#ifdef ARCANE_LIMA_HAS_MLI2
#include <Lima/malipp2.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LimaUtils::
createGroup(IItemFamily* family,const String& name,Int32ArrayView local_ids)
{
  ITraceMng* tm = family->traceMng();
  if (!local_ids.empty())
    std::sort(std::begin(local_ids),std::end(local_ids));
  Integer nb_item = local_ids.size();
  Integer nb_duplicated = 0;
  // Détecte les doublons
  for( Integer i=1; i<nb_item; ++i )
    if (local_ids[i]==local_ids[i-1]){
      ++nb_duplicated;
    }
  if (nb_duplicated!=0){
    tm->warning() << "Duplicated items in group name=" << name
                  << " nb_duplicated=" << nb_duplicated;
    auto xbegin = std::begin(local_ids);
    auto xend = std::end(local_ids);
    Integer new_size = CheckedConvert::toInteger(std::unique(xbegin,xend)-xbegin);
    tm->info() << "NEW_SIZE=" << new_size << " old=" << nb_item;
    local_ids = local_ids.subView(0,new_size);
  }

  family->createGroup(name,local_ids,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType
LimaUtils::_directLimaPartitionMalipp(ITimerMng* timer_mng,IPrimaryMesh* mesh,
                           const String& filename, Real length_multiplier)
{
#ifdef ARCANE_LIMA_HAS_MLI
  LimaMalippReader<Lima::MaliPPReader> reader(mesh->traceMng());
  IMeshReader::eReturnType rt = reader.readMeshFromFile(timer_mng,mesh, filename, length_multiplier);
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

IMeshReader::eReturnType
LimaUtils::_directLimaPartitionMalipp2(ITimerMng* timer_mng,IPrimaryMesh* mesh,
                           const String& filename, Real length_multiplier)
{
#ifdef ARCANE_LIMA_HAS_MLI2
  LimaMalippReader<Lima::MaliPPReader2> reader(mesh->traceMng());
  auto rt = reader.readMeshFromFile(timer_mng,mesh,filename,length_multiplier);
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
