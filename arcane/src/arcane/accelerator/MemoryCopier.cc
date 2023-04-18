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

  void copyFrom(Span<const Int32> indexes, Span<const std::byte> source,
                Span<std::byte> destination) override
  {
    _copyFrom(indexes, _toTrueType(source), _toTrueType(destination));
  }

  void copyTo(Span<const Int32> indexes, Span<const std::byte> source,
              Span<std::byte> destination) override
  {
    _copyTo(indexes, _toTrueType(source), _toTrueType(destination));
  }

 public:

  void _copyFrom(Span<const Int32> indexes, Span<const DataType> source,
                 Span<DataType> destination)
  {
    Int32 nb_index = (Int32)indexes.size();

    auto command = makeCommand(m_queue);
    command << RUNCOMMAND_LOOP1(iter, nb_index)
    {
      auto [i] = iter();
      Int64 zindex = i * m_extent.v;
      Int64 zci = indexes[i] * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[zindex + z] = source[zci + z];
    };
  }

  void _copyTo(Span<const Int32> indexes, Span<const DataType> source,
               Span<DataType> destination)
  {
    Int32 nb_index = (Int32)indexes.size();

    auto command = makeCommand(m_queue);
    command << RUNCOMMAND_LOOP1(iter, nb_index)
    {
      auto [i] = iter();
      Int64 zindex = i * m_extent.v;
      Int64 zci = indexes[i] * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[zci + z] = source[zindex + z];
    };
  }

 private:

  RunQueue* m_queue = nullptr;
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

namespace
{
  Arcane::impl::SpecificMemoryCopyList<AcceleratorIndexedCopyTraits> global_copy_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
