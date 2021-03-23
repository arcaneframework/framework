// -*- coding: utf-8 -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCANE_STD_TYPESMETISMESHPARTITIONER
#define ARCANE_STD_TYPESMETISMESHPARTITIONER
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "arcane/utils/ArcaneGlobal.h"
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
ARCANE_BEGIN_NAMESPACE
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class TypesMetisMeshPartitioner
{
  public:
  
  enum class MetisCallStrategy
  { 
    all_processors,
    one_processor_per_node,
    two_processors_two_nodes,
    two_gathered_processors,
    two_scattered_processors
  };
  enum class MetisEmptyPartitionStrategy
  {
    DoNothing = 0,
    TakeFromBiggestPartitionV1,
    TakeFromBiggestPartitionV2
  };
};
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
ARCANE_END_NAMESPACE
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif
