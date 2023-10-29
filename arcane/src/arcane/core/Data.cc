// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Data.cc                                                     (C) 2000-2023 */
/*                                                                           */
/* Types liés aux 'IData'.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IData.h"

#include "arcane/utils/String.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IDataFactory.h"
#include "arcane/core/IDataStorageFactory.h"
#include "arcane/core/IDataVisitor.h"
#include "arcane/core/ISerializedData.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/Memory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::IData);
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::ISerializedData);
} // namespace Arccore

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IDataVisitor::
applyDataVisitor(IMultiArray2Data*)
{
  ARCANE_THROW(NotSupportedException, "using applyDataVisitor with IMultiArray2Data is no longer supported");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IData::
visitMultiArray2(IMultiArray2DataVisitor*)
{
  ARCANE_THROW(NotSupportedException, "Visiting IMultiArray2Data is no longer supported");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::
copyContigousData(IData* destination, IData* source, RunQueue& queue)
{
  ARCANE_CHECK_POINTER(source);
  ARCANE_CHECK_POINTER(destination);

  INumericDataInternal* num_source = destination->_commonInternal()->numericData();
  if (!num_source)
    ARCANE_FATAL("Source is not a numerical data");
  INumericDataInternal* num_destination = source->_commonInternal()->numericData();
  if (!num_destination)
    ARCANE_FATAL("Destination is not a numerical data");
  MutableMemoryView destination_buf = num_destination->memoryView();
  ConstMemoryView source_buf = num_source->memoryView();
  if (source_buf.datatypeSize() != destination_buf.datatypeSize())
    ARCANE_FATAL("Source and destination do not have the same datatype s={0} d={1}",
                 source_buf.datatypeSize(), destination_buf.datatypeSize());
  queue.copyMemory(Accelerator::MemoryCopyArgs(destination_buf, source_buf));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
