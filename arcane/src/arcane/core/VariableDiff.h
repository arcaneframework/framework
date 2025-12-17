// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableDiff.h                                              (C) 2000-2025 */
/*                                                                           */
/* Gestion des différences entre les variables                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEDIFF_H
#define ARCANE_CORE_VARIABLEDIFF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/core/VariableDataTypeTraits.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Classe de base pour les comparaisons de valeurs entre deux variables.
 */
template <typename DataType>
class VariableDiff
{
 public:

  using VarDataTypeTraits = VariableDataTypeTraitsT<DataType>;
  static constexpr bool IsNumeric = std::is_same_v<typename VarDataTypeTraits::IsNumeric, TrueType>;

 public:

  class DiffInfo
  {
   public:

    using VarDataTypeTraits = VariableDataTypeTraitsT<DataType>;

   public:

    DiffInfo() = default;
    DiffInfo(const DataType& current, const DataType& ref, const DataType& diff,
             Item item, Integer sub_index)
    : m_current(current)
    , m_ref(ref)
    , m_diff(diff)
    , m_sub_index(sub_index)
    , m_is_own(item.isOwn())
    , m_local_id(item.localId())
    , m_unique_id(item.uniqueId())
    {}
    DiffInfo(const DataType& current, const DataType& ref, const DataType& diff,
             Int32 index, Integer sub_index)
    : m_current(current)
    , m_ref(ref)
    , m_diff(diff)
    , m_sub_index(sub_index)
    , m_is_own(false)
    , m_local_id(index)
    , m_unique_id(NULL_ITEM_UNIQUE_ID)
    {}

   public:

    DataType m_current = {};
    DataType m_ref = {};
    DataType m_diff = {};
    Integer m_sub_index = NULL_ITEM_ID;
    bool m_is_own = false;
    Int32 m_local_id = NULL_ITEM_LOCAL_ID;
    Int64 m_unique_id = NULL_ITEM_UNIQUE_ID;

   public:

    bool operator<(const DiffInfo& t2) const
    {
      return VarDataTypeTraits::normeMax(m_diff) > VarDataTypeTraits::normeMax(t2.m_diff);
    }
  };

  UniqueArray<DiffInfo> m_diffs_info;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  void sort(TrueType)
  {
    _sort();
  }

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  void sort(FalseType)
  {
  }

 protected:

  void _sortAndDump(IVariable* var, IParallelMng* pm, const VariableComparerArgs& compare_args)
  {
    _sort();
    dump(var, pm, compare_args);
  }

  void dump(IVariable* var, IParallelMng* pm, const VariableComparerArgs& compare_args)
  {
    DiffPrinter::dump(m_diffs_info, var, pm, compare_args);
  }
  void _sort()
  {
    DiffPrinter::sort(m_diffs_info);
  }

 private:

  class DiffPrinter
  {
   public:

    ARCANE_CORE_EXPORT static void
    dump(ConstArrayView<DiffInfo> diff_infos, IVariable* var, IParallelMng* pm,
         const VariableComparerArgs& compare_args);
    ARCANE_CORE_EXPORT static void sort(ArrayView<DiffInfo> diff_infos);
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
