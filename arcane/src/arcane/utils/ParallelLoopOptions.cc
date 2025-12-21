// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelLoopOptions.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Options de configuration pour les boucles parallèles en multi-thread.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ParallelLoopOptions.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/internal/Property.h"
#include "arcane/utils/internal/ParallelLoopOptionsProperties.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  const char*
  _partitionerToString(ParallelLoopOptions::Partitioner p)
  {
    switch (p) {
    case ParallelLoopOptions::Partitioner::Static:
      return "static";
    case ParallelLoopOptions::Partitioner::Deterministic:
      return "deterministic";
    case ParallelLoopOptions::Partitioner::Auto:
      return "auto";
    }
    ARCANE_FATAL("Bad value {0} for partitioner", (int)p);
  }

  ParallelLoopOptions::Partitioner
  _stringToPartitioner(const String& str)
  {
    if (str == "static")
      return ParallelLoopOptions::Partitioner::Static;
    if (str == "deterministic")
      return ParallelLoopOptions::Partitioner::Deterministic;
    if (str == "auto")
      return ParallelLoopOptions::Partitioner::Auto;
    ARCANE_FATAL("Bad value '{0}' for partitioner. Valid values are 'auto', 'static' or 'deterministic'", str);
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename V> void ParallelLoopOptionsProperties::
_applyPropertyVisitor(V& p)
{
  auto b = p.builder();
  p << b.addInt32("ParallelLoopGrainSize")
       .addDescription("GrainSize of the loop")
       .addCommandLineArgument("ParallelLoopGrainSize")
       .addGetter([](auto a) { return a.x.grainSize(); })
       .addSetter([](auto a) { a.x.setGrainSize(a.v); });

  p << b.addString("ParallelLoopPartitioner")
       .addDescription("Partitioner for the loop (auto, static or deterministic)")
       .addCommandLineArgument("ParallelLoopPartitioner")
       .addGetter([](auto a) { return _partitionerToString(a.x.partitioner()); })
       .addSetter([](auto a) { a.x.setPartitioner(_stringToPartitioner(a.v)); });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_PROPERTY_CLASS(ParallelLoopOptionsProperties, ());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
