// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephOrdering.h                                             (C) 2000-2024 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ALEPH_ALEPHORDERING_H
#define ARCANE_ALEPH_ALEPHORDERING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de reordering
 */
class AlephOrdering
: public TraceAccessor
{
 public:

  explicit AlephOrdering(AlephKernel*);
  AlephOrdering(AlephKernel*, Integer, Integer, bool = false);
  ~AlephOrdering();

 public:

  Integer swap(Integer i)
  {
    if (m_do_swap)
      return CheckedConvert::toInteger(m_swap.at(i));
    return i;
  }

 private:

  void initCellOrder();
  void initTwiceCellOrder();
  void initFaceOrder();
  void initCellFaceOrder();
  void initCellNodeOrder();
  void initTwiceCellNodeOrder();

 private:

  bool m_do_swap = false;
  AlephKernel* m_kernel = nullptr;
  UniqueArray<Int64> m_swap;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
