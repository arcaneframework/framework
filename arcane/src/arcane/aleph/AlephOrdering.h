// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephOrdering.h                                                  (C) 2012 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_ORDERING_H
#define ALEPH_ORDERING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionaire de reordering
 */
class AlephOrdering
: public TraceAccessor
{
 public:
  AlephOrdering(AlephKernel*);
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
  void initCellOrder(void);
  void initTwiceCellOrder(void);
  void initFaceOrder(void);
  void initCellFaceOrder(void);
  void initCellNodeOrder(void);
  void initTwiceCellNodeOrder(void);

 private:
  bool m_do_swap;
  AlephKernel* m_kernel;

 private:
  UniqueArray<Int64> m_swap;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
