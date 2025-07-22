// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Data.cc                                                     (C) 2000-2025 */
/*                                                                           */
/* Types liés aux 'IData'.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IData.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/IDataFactory.h"
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
copyContiguousData(INumericDataInternal* num_destination, ConstMemoryView source_buf,
                   RunQueue& queue)
{
  ARCANE_CHECK_POINTER(num_destination);

  MutableMemoryView destination_buf = num_destination->memoryView();
  if (source_buf.datatypeSize() != destination_buf.datatypeSize())
    ARCANE_FATAL("Source and destination do not have the same datatype s={0} d={1}",
                 source_buf.datatypeSize(), destination_buf.datatypeSize());
  if (queue.isNull())
    MemoryUtils::copy(destination_buf, source_buf);
  else
    queue.copyMemory(Accelerator::MemoryCopyArgs(destination_buf, source_buf));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::
copyContiguousData(IData* destination, IData* source, RunQueue& queue)
{
  ARCANE_CHECK_POINTER(source);
  ARCANE_CHECK_POINTER(destination);

  INumericDataInternal* num_destination = destination->_commonInternal()->numericData();
  if (!num_destination)
    ARCANE_FATAL("Destination is not a numerical data");
  INumericDataInternal* num_source = source->_commonInternal()->numericData();
  if (!num_source)
    ARCANE_FATAL("Source is not a numerical data");
  copyContiguousData(num_destination, num_source->memoryView(), queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::
fillContiguousDataGeneric(IData* data, const void* fill_address,
                          Int32 datatype_size, RunQueue& queue)
{
  INumericDataInternal* num_data = data->_commonInternal()->numericData();
  if (!num_data)
    ARCANE_FATAL("Destination is not a numerical data");

  ConstMemoryView fill_value_view(makeConstMemoryView(fill_address, datatype_size, 1));
  MutableMemoryView destination_buf = num_data->memoryView();

  // Si \a data est un tableau 2D ou plus il faut le transformer en un tableau 1D
  // du nombre total d'éléments sinon 'destination_buf.datatypeSize() n'est pas
  // cohérent avec 'datatype_size'
  if (data->dimension() > 1) {
    Int64 total_dim = data->shape().totalNbElement();
    Int64 nb_element = destination_buf.nbElement();
    destination_buf = makeMutableMemoryView(destination_buf.data(), datatype_size, nb_element * total_dim);
  }

  MemoryUtils::fill(destination_buf, fill_value_view, &queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
