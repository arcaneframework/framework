﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPrinter.h                                               (C) 2000-2004 */
/*                                                                           */
/* Routines d'impressions d'une entité.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMPRINTER_H
#define ARCANE_ITEMPRINTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/Iostream.h"

#include "arcane/ArcaneTypes.h"

#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe utilitaire pour imprimer les infos sur une entité.
 */
class ARCANE_CORE_EXPORT ItemPrinter
{
 public:
  ItemPrinter(ItemInternal* item,eItemKind ik) 
  : m_item(item), m_item_kind(ik), m_has_item_kind(true)
    {}
  ItemPrinter(ItemInternal* item)
  : m_item(item), m_item_kind(IK_Unknown), m_has_item_kind(false)
    {}
  ItemPrinter(const Item& item)
  : m_item(item.internal()), m_item_kind(IK_Unknown), m_has_item_kind(false)
    {}
  ItemPrinter(const Item& item,eItemKind ik)
  : m_item(item.internal()), m_item_kind(ik), m_has_item_kind(true)
    {}
  ItemPrinter(const Node& item)
  : m_item(item.internal()), m_item_kind(IK_Node), m_has_item_kind(true)
    {}
  ItemPrinter(const Edge& item)
  : m_item(item.internal()), m_item_kind(IK_Edge), m_has_item_kind(true)
    {}
  ItemPrinter(const Face& item)
  : m_item(item.internal()), m_item_kind(IK_Face), m_has_item_kind(true)
    {}
  ItemPrinter(const Cell& item)
  : m_item(item.internal()), m_item_kind(IK_Cell), m_has_item_kind(true)
    {}
  ItemPrinter(const Particle& item)
  : m_item(item.internal()), m_item_kind(IK_Particle), m_has_item_kind(true)
    {}
 public:

  //! Ecriture sur flux de l'Item courant
  void print(std::ostream& o) const;

 public:
  ItemInternal* m_item;
  eItemKind m_item_kind;
  bool m_has_item_kind;

 public:
  struct Internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT FullItemPrinter {
public:
  explicit FullItemPrinter(const Item& item)
    : m_item(item.internal())
  {}
 
  //! Ecriture sur flux de l'Item courant et de ses sous-items
  void print(std::ostream& o) const;
  
private:
   ItemInternal* m_item;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT NeighborItemPrinter
{
 public:
  explicit NeighborItemPrinter(Item item, const Integer levelmax=1)
  : m_item(item.internal()), m_level_max(levelmax)
  {}
 
  //! Ecriture sur flux de l'Item courant et de ses sous-items
  void print(std::ostream& o) const { print(o,m_item,m_level_max,m_level_max); }

 private:
   ItemInternal* m_item;
   Integer m_level_max;
   static std::ostream& indent(std::ostream& o, Integer n);
   static void print(std::ostream& o, ItemInternal * item, Integer level, Integer levelmax);
   static void _printSubItems(std::ostream& ostr,Integer level, Integer levelmax,
                              ItemInternalVectorView sub_items,const char* name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o,const ItemPrinter& ip)
{
  ip.print(o);
  return o;
}

inline std::ostream&
operator<<(std::ostream& o,const FullItemPrinter& ip)
{
  ip.print(o);
  return o;
}

inline std::ostream&
operator<<(std::ostream& o,const NeighborItemPrinter& ip)
{
  ip.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
