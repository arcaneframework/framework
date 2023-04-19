// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryCopier.cc                                             (C) 2000-2023 */
/*                                                                           */
/* Fonctions diverses de copie mémoire.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/internal/SpecificMemoryCopyList.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{
using IndexedMemoryCopyArgs = Arcane::impl::IndexedMemoryCopyArgs;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename Extent>
class AcceleratorSpecificMemoryCopy
: public Arcane::impl::SpecificMemoryCopyBase<DataType, Extent>
{
  using BaseClass = Arcane::impl::SpecificMemoryCopyBase<DataType, Extent>;
  using BaseClass::_toTrueType;

 public:

  using BaseClass::m_extent;

 public:

  void copyFrom(const IndexedMemoryCopyArgs& args) override
  {
    _copyFrom(args.m_queue, args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }

  void copyTo(const IndexedMemoryCopyArgs& args) override
  {
    _copyTo(args.m_queue, args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }

 public:

  void _copyFrom(RunQueue* queue, Span<const Int32> indexes,
                 Span<const DataType> source, Span<DataType> destination)
  {
    ARCANE_CHECK_POINTER(queue);

    Int32 nb_index = (Int32)indexes.size();
    const Int32 sub_size = m_extent.v;

    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(iter, nb_index)
    {
      auto [i] = iter();
      Int64 zindex = i * sub_size;
      Int64 zci = indexes[i] * sub_size;
      for (Int32 z = 0; z < sub_size; ++z)
        destination[zindex + z] = source[zci + z];
    };
  }

  void _copyTo(RunQueue* queue, Span<const Int32> indexes, Span<const DataType> source,
               Span<DataType> destination)
  {
    ARCANE_CHECK_POINTER(queue);

    Int32 nb_index = (Int32)indexes.size();
    const Int32 sub_size = m_extent.v;

    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(iter, nb_index)
    {
      auto [i] = iter();
      Int64 zindex = i * sub_size;
      Int64 zci = indexes[i] * sub_size;
      for (Int32 z = 0; z < sub_size; ++z)
        destination[zci + z] = source[zindex + z];
    };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AcceleratorIndexedCopyTraits
{
 public:

  using InterfaceType = Arcane::impl::ISpecificMemoryCopy;
  template <typename DataType, typename Extent> using SpecificType = AcceleratorSpecificMemoryCopy<DataType, Extent>;
  using RefType = Arcane::impl::SpecificMemoryCopyRef<AcceleratorIndexedCopyTraits>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AcceleratorSpecificMemoryCopyList
{
 public:
  AcceleratorSpecificMemoryCopyList()
  {
    Arcane::impl::ISpecificMemoryCopyList::setDefaultCopyListIfNotSet(&m_copy_list);
  }
  Arcane::impl::SpecificMemoryCopyList<AcceleratorIndexedCopyTraits> m_copy_list;
};

namespace
{
  AcceleratorSpecificMemoryCopyList global_copy_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
