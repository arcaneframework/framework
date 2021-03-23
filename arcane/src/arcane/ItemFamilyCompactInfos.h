// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyCompactInfos.h                                    (C) 2000-2016 */
/*                                                                           */
/* Informations pour gérer le compactage des entités d'une famille.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMFAMILYCOMPACTINFOS_H
#define ARCANE_ITEMFAMILYCOMPACTINFOS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshCompacter;
class IItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour gérer le compactage des entités d'une famille.
 */
class ARCANE_CORE_EXPORT ItemFamilyCompactInfos
{
 public:
  ItemFamilyCompactInfos(IMeshCompacter* acompacter,IItemFamily* afamily)
  : m_compacter(acompacter), m_family(afamily){}
  ~ItemFamilyCompactInfos() {}
 public:
  IMeshCompacter* compacter() const { return m_compacter; }
  IItemFamily* family() const { return m_family; }
  //! Conversion entre les anciens et les nouveaux id locaux.
  Int32ConstArrayView oldToNewLocalIds() const
  {
    return m_old_to_new_local_ids;
  }

  //! Conversion entre les nouveaux et les anciens id locaux.
  Int32ConstArrayView newToOldLocalIds() const
  {
    return m_new_to_old_local_ids;
  }
  void setOldToNewLocalIds(UniqueArray<Int32>&& ids)
  {
    m_old_to_new_local_ids = ids;
  }
  void setNewToOldLocalIds(UniqueArray<Int32>&& ids)
  {
    m_new_to_old_local_ids = ids;
  }
  void clear()
  {
    m_old_to_new_local_ids.clear();
    m_new_to_old_local_ids.clear();
  }
 private:
  IMeshCompacter* m_compacter;
  IItemFamily* m_family;
  UniqueArray<Int32> m_old_to_new_local_ids;
  UniqueArray<Int32> m_new_to_old_local_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
