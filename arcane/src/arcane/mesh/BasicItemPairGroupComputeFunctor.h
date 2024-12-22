// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicItemPairGroupComputeFunctor.h                          (C) 2000-2024 */
/*                                                                           */
/* Fonctions utilitaires sur un maillage.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_BASICITEMPAIRGROUPCOMPUTEFUNCTOR_H
#define ARCANE_MESH_BASICITEMPAIRGROUPCOMPUTEFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/Item.h"

#include <map>
#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemPairGroup;
class ItemPairGroupImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions utilitaires sur un maillage.
 */
class BasicItemPairGroupComputeFunctor
: public TraceAccessor
{
 public:

  struct AdjacencyType
  {
    AdjacencyType()
    : m_item_kind(IK_Unknown), m_sub_item_kind(IK_Unknown),
      m_link_item_kind(IK_Unknown)
    {
    }
    AdjacencyType(eItemKind ik, eItemKind sik, eItemKind lik)
    : m_item_kind(ik), m_sub_item_kind(sik), m_link_item_kind(lik)
    {
    }
    eItemKind m_item_kind;
    eItemKind m_sub_item_kind;
    eItemKind m_link_item_kind;
    bool operator<(const AdjacencyType& rhs) const
    {
      if (m_item_kind != rhs.m_item_kind)
        return m_item_kind < rhs.m_item_kind;
      if (m_sub_item_kind != rhs.m_sub_item_kind)
        return m_sub_item_kind < rhs.m_sub_item_kind;
      return m_link_item_kind < rhs.m_link_item_kind;
    }
  };

  typedef void (BasicItemPairGroupComputeFunctor::*ComputeFunctor)(ItemPairGroupImpl* array);

  class AdjacencyComputeFunctor
  : public IFunctor
  {
   public:

    AdjacencyComputeFunctor(BasicItemPairGroupComputeFunctor* ptr,
                            ItemPairGroupImpl* array, ComputeFunctor func_ptr)
    : m_ptr(ptr), m_array(array), m_func_ptr(func_ptr) {}
   public:

    void executeFunctor() override
    {
      (m_ptr->*m_func_ptr)(m_array);
    }
   private:
    BasicItemPairGroupComputeFunctor* m_ptr;
    ItemPairGroupImpl* m_array;
    ComputeFunctor m_func_ptr;
  };

 public:

  explicit BasicItemPairGroupComputeFunctor(ITraceMng* tm);

 public:

  virtual void computeAdjacency(ItemPairGroup adjacency_array, eItemKind link_kind,
                                Integer nb_layer);

 private:

  std::map<AdjacencyType, ComputeFunctor> m_compute_adjacency_functions;

 private:

  void _addComputeAdjacency(eItemKind ik, eItemKind sik, eItemKind lik, ComputeFunctor f);
  void _computeCellCellNodeAdjacency(ItemPairGroupImpl* array);
  void _computeCellCellFaceAdjacency(ItemPairGroupImpl* array);
  void _computeNodeNodeCellAdjacency(ItemPairGroupImpl* array);
  void _computeFaceCellNodeAdjacency(ItemPairGroupImpl* array);
  void _computeFaceFaceNodeAdjacency(ItemPairGroupImpl* array);
  void _computeCellFaceFaceAdjacency(ItemPairGroupImpl* array);
  void _computeNodeNodeFaceAdjacency(ItemPairGroupImpl* array);
  void _computeNodeNodeEdgeAdjacency(ItemPairGroupImpl* array);
  void _computeFaceFaceEdgeAdjacency(ItemPairGroupImpl* array);
  void _computeFaceFaceCellAdjacency(ItemPairGroupImpl* array);

  using GetItemVectorViewFunctor = std::function<ItemConnectedListViewType(Item)>;
  void _computeAdjacency(ItemPairGroupImpl* array, GetItemVectorViewFunctor get_item_enumerator,
                         GetItemVectorViewFunctor get_sub_item_enumerator);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
