// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableDiff.h                                              (C) 2000-2023 */
/*                                                                           */
/* Gestion des différences entre les variables                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLEDIFF_H
#define ARCANE_VARIABLEDIFF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/datatype/DataTypes.h"

#include "arcane/core/VariableDataTypeTraits.h"
#include "arcane/core/Item.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IParallelMng.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Classe de base pour les comparaisons de valeurs entre deux variables.
 */
template<typename DataType>
class VariableDiff
{
 public:
  typedef VariableDataTypeTraitsT<DataType> VarDataTypeTraits;
 public:
  class DiffInfo
  {
  public:
    typedef VariableDataTypeTraitsT<DataType> VarDataTypeTraits;
  public:
    DiffInfo()
    : m_current(DataType()), m_ref(DataType()), m_diff(DataType()),
      m_sub_index(NULL_ITEM_ID), m_is_own(false), m_local_id(NULL_ITEM_LOCAL_ID),
      m_unique_id(NULL_ITEM_UNIQUE_ID){}
    DiffInfo(const DataType& current,const DataType& ref,const DataType& diff,
             Item item,Integer sub_index)
    : m_current(current), m_ref(ref), m_diff(diff),
      m_sub_index(sub_index), m_is_own(item.isOwn()), m_local_id(item.localId()),
      m_unique_id(item.uniqueId()){}
    DiffInfo(const DataType& current,const DataType& ref,const DataType& diff,
             Int32 index,Integer sub_index)
    : m_current(current), m_ref(ref), m_diff(diff),
      m_sub_index(sub_index), m_is_own(false), m_local_id(index),
      m_unique_id(NULL_ITEM_UNIQUE_ID){}
  public:
    DataType m_current;
    DataType m_ref;
    DataType m_diff;
    Integer m_sub_index;
    bool m_is_own;
    Int32 m_local_id;
    Int64 m_unique_id;
  public:
    bool operator<(const DiffInfo& t2) const
    {
      return VarDataTypeTraits::normeMax(m_diff) > VarDataTypeTraits::normeMax(t2.m_diff);
    }
  };
  UniqueArray<DiffInfo> m_diffs_info;

 public:
  
  void sort(TrueType)
  {
    std::sort(std::begin(m_diffs_info),std::end(m_diffs_info));
  }
  void sort(FalseType)
  {
  }

 public:

 protected:

  void dump(IVariable* var,IParallelMng* pm, int max_print)
  {
    ITraceMng* msg = pm->traceMng();
    Int32 sid = pm->commRank();
    const String& var_name = var->name();
    Integer nb_diff = m_diffs_info.size();
    Integer nb_print = nb_diff;
    if (max_print>=0 && nb_diff>static_cast<Integer>(max_print))
      nb_print = max_print;
    OStringStream ostr;
    ostr().precision(FloatInfo<Real>::maxDigit());
    ostr() << nb_diff << " entities having different values for the variable "
           << var_name << '\n';
    for( Integer i=0; i<nb_print; ++i ){
      const DiffInfo& di = m_diffs_info[i];
      if (di.m_unique_id!=NULL_ITEM_UNIQUE_ID){
        // Il s'agit d'une entité
        char type = di.m_is_own ? 'O' : 'G';
        ostr() << "VDIFF: Variable '" << var_name << "'"
               << " (" << type << ")"
               << " uid=" << di.m_unique_id
               << " lid=" << di.m_local_id;
        if (di.m_sub_index!=NULL_ITEM_ID)
          ostr() << " [" << di.m_sub_index << "]";
        ostr() << " val: " << di.m_current
               << " ref: " << di.m_ref << " rdiff: " << di.m_diff << '\n';
      }
      else{
        // Il s'agit de l'indice d'une variable tableau
        ostr() << "VDIFF: Variable '" << var_name << "'"
               << " index=" << di.m_local_id;
        if (di.m_sub_index!=NULL_ITEM_ID)
          ostr() << " [" << di.m_sub_index << "]";
        ostr() << " val: " << di.m_current
               << " ref: " << di.m_ref << " rdiff: " << di.m_diff << '\n';
      }
    }
    msg->pinfo() << "Processor " << sid << " : " << nb_diff
                 << " values are different on the variable "
                 << var_name << ":\n" << ostr.str();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

