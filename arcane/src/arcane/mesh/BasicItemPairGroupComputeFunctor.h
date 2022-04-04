// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicItemPairGroupComputeFunctor.h                          (C) 2000-2016 */
/*                                                                           */
/* Fonctions utilitaires sur un maillage.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_BASICITEMPAIRGROUPCOMPUTEFUNCTOR_H
#define ARCANE_MESH_BASICITEMPAIRGROUPCOMPUTEFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/IMeshUtilities.h"

#include <map>
#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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
  
  struct AdjencyType
  {
    AdjencyType()
    : m_item_kind(IK_Unknown), m_sub_item_kind(IK_Unknown),
      m_link_item_kind(IK_Unknown)
    {
    }
    AdjencyType(eItemKind ik,eItemKind sik,eItemKind lik)
    : m_item_kind(ik), m_sub_item_kind(sik), m_link_item_kind(lik)
    {
    }
    eItemKind m_item_kind;
    eItemKind m_sub_item_kind;
    eItemKind m_link_item_kind;
    bool operator<(const AdjencyType& rhs) const
    {
      if (m_item_kind != rhs.m_item_kind)
        return m_item_kind < rhs.m_item_kind;
      if (m_sub_item_kind != rhs.m_sub_item_kind)
        return m_sub_item_kind < rhs.m_sub_item_kind;
      return m_link_item_kind < rhs.m_link_item_kind;
    }
  };

  typedef void (BasicItemPairGroupComputeFunctor::*ComputeFunctor)(ItemPairGroupImpl* array);

  class AdjencyComputeFunctor
  : public IFunctor
  {
   public:
    AdjencyComputeFunctor(BasicItemPairGroupComputeFunctor* ptr,
                          ItemPairGroupImpl* array,ComputeFunctor func_ptr)
    : m_ptr(ptr), m_array(array), m_func_ptr(func_ptr) {}
   public:
    virtual void executeFunctor()
    {
      (m_ptr->*m_func_ptr)(m_array);
    }
   private:
    BasicItemPairGroupComputeFunctor* m_ptr;
    ItemPairGroupImpl* m_array;
    ComputeFunctor m_func_ptr;
  };

 public:

  BasicItemPairGroupComputeFunctor(ITraceMng* tm);
  virtual ~BasicItemPairGroupComputeFunctor() {} //!< Libère les ressources.

 public:

  virtual void computeAdjency(ItemPairGroup adjency_array,eItemKind link_kind,
                              Integer nb_layer);


 private:

  IMesh* m_mesh;
  std::map<AdjencyType,ComputeFunctor> m_compute_adjency_functions;

 private:

 private:
  void _addComputeAdjency(eItemKind ik,eItemKind sik,eItemKind lik,ComputeFunctor f);
  void _computeCellCellNodeAdjency(ItemPairGroupImpl* array);
  void _computeCellCellFaceAdjency(ItemPairGroupImpl* array);
  void _computeNodeNodeCellAdjency(ItemPairGroupImpl* array);
  void _computeFaceCellNodeAdjency(ItemPairGroupImpl* array);
  void _computeFaceFaceNodeAdjency(ItemPairGroupImpl* array);
  void _computeCellFaceFaceAdjency(ItemPairGroupImpl* array);
  void _computeNodeNodeFaceAdjency(ItemPairGroupImpl* array);
  void _computeNodeNodeEdgeAdjency(ItemPairGroupImpl* array);
  void _computeFaceFaceEdgeAdjency(ItemPairGroupImpl* array);
  void _computeFaceFaceCellAdjency(ItemPairGroupImpl* array);

  typedef std::function<ItemVectorView(Item)> GetItemVectorViewFunctor;
  void _computeAdjency(ItemPairGroupImpl* array,GetItemVectorViewFunctor get_item_enumerator,
                       GetItemVectorViewFunctor get_sub_item_enumerator);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

