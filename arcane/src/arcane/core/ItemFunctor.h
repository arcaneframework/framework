// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFunctor.h                                               (C) 2000-2024 */
/*                                                                           */
/* Functor on entities.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMFUNCTOR_H
#define ARCANE_ITEMFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/RangeFunctor.h"
#include "arcane/utils/Functor.h"

#include "arcane/core/Item.h"
#include "arcane/core/ItemVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for functors on a list of entities.
 *
 * This class allows splitting an iteration over an ItemVector by
 * ensuring that iterations occur on a multiple of \a m_block_size.
 * For now, this value is always 8, and thus iterations over
 * entities are done in blocks of 8 values. This allows guaranteeing for
 * vectorization that the sub-views of \a m_items will be correctly aligned.
 */
class ARCANE_CORE_EXPORT AbstractItemRangeFunctor
: public IRangeFunctor
{
 public:

  static const Integer DEFAULT_GRAIN_SIZE = 400;

  AbstractItemRangeFunctor(ItemVectorView items_view, Int32 grain_size);

 public:

  //! Number of blocks.
  Int32 nbBlock() const { return m_nb_block; }

  //! Desired size of an iteration interval.
  Int32 blockGrainSize() const { return m_block_grain_size; }

 protected:

  ItemVectorView m_items;
  Int32 m_block_size = 0;
  Int32 m_nb_block = 0;
  Int32 m_block_grain_size = 0;

 protected:

  ItemVectorView _view(Int32 begin_block, Int32 nb_block, Int32* true_begin = nullptr) const;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Functor for iterating over a list of entities.
 */
template <typename InstanceType, typename ItemType>
class ItemRangeFunctorT
: public AbstractItemRangeFunctor
{
 private:

  typedef void (InstanceType::*FunctionType)(ItemVectorViewT<ItemType>);

 public:

  ItemRangeFunctorT(ItemVectorView items_view, InstanceType* instance,
                    FunctionType function, Integer grain_size = DEFAULT_GRAIN_SIZE)
  : AbstractItemRangeFunctor(items_view, grain_size)
  , m_instance(instance)
  , m_function(function)
  {
  }

 private:

  InstanceType* m_instance;
  FunctionType m_function;

 public:

  virtual void executeFunctor(Int32 begin, Int32 size)
  {
    //cout << "** BLOCKED RANGE! range=" << range.begin() << " end=" << range.end() << " size=" << range.size() << "\n";
    //CellVectorView sub_view = m_cells.subView(range.begin(),range.size());
    ItemVectorViewT<ItemType> sub_view(this->_view(begin, size));
    //cout << "** SUB_VIEW v=" << sub_view.size();
    (m_instance->*m_function)(sub_view);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Functor on an iteration interval instantiated via a lambda function.
 *
 * This class is used with the C++1x lambda function mechanism.
 */
template <typename LambdaType>
class LambdaItemRangeFunctorT
: public AbstractItemRangeFunctor
{
 public:

  LambdaItemRangeFunctorT(ItemVectorView items_view, const LambdaType& lambda_function,
                          Int32 grain_size = DEFAULT_GRAIN_SIZE)
  : AbstractItemRangeFunctor(items_view, grain_size)
  , m_lambda_function(lambda_function)
  {
  }

 public:

  void executeFunctor(Int32 begin, Int32 size) override
  {
    Int32 true_begin = 0;
    ItemVectorView sub_view(this->_view(begin, size, &true_begin));
    // The lambda can have two prototypes:
    // - it takes only an ItemVectorView as argument (historical version)
    // - it takes an ItemVectorView and the starting index of the vector. This
    // allows knowing the iteration index
    if constexpr (std::is_invocable_v<LambdaType, ItemVectorView>)
      m_lambda_function(sub_view);
    else
      m_lambda_function(sub_view, true_begin);
  }

 private:

  const LambdaType& m_lambda_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Functor for calculating elements of a group.
 */
class ItemGroupComputeFunctor
: public IFunctor
{
 public:

  ItemGroupComputeFunctor() = default;

 public:

  void setGroup(ItemGroupImpl* group) { m_group = group; }

 protected:

  ItemGroupImpl* m_group = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
