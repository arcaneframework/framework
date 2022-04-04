// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephIndexing.h                                             (C) 2012~2013 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_INDEXING_H
#define ALEPH_INDEXING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Gestionaire d'indexing
 */
class ARCANE_ALEPH_EXPORT AlephIndexing : public TraceAccessor
{
 public:
  AlephIndexing(AlephKernel*);
  ~AlephIndexing();

 public:
  Int32 updateKnownItems(VariableItemInt32*, const Item&);
  Int32 findWhichLidFromMapMap(IVariable*, const Item&);
  Integer get(const VariableRef&, const ItemEnumerator&);
  Integer get(const VariableRef&, const Item&);
  void buildIndexesFromAddress(void);
  void nowYouCanBuildTheTopology(AlephMatrix*, AlephVector*, AlephVector*);

 private:
  Integer localKnownItems(void);

 private:
  AlephKernel* m_kernel;
  ISubDomain* m_sub_domain;
  Integer m_current_idx;
  Int32 m_known_items_own;
  UniqueArray<Int32*> m_known_items_all_address;
  typedef std::map<IVariable*, VariableItemInt32*> VarMapIdx;
  VarMapIdx m_var_map_idx;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
