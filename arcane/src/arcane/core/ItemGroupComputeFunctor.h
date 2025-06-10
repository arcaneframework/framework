// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupComputeFunctor.h                                   (C) 2000-2025 */
/*                                                                           */
/* Functors de calcul des éléments d'un groupe en fonction d'un autre groupe */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMGROUPCOMPUTEFUNCTOR_H
#define ARCANE_CORE_ITEMGROUPCOMPUTEFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemFunctor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * NOTE: ce fichier est interne à Arcane. A terme, il sera dans
 * arcane/impl.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OwnItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GhostItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class InterfaceItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType>
class ItemItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class InnerFaceItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OuterFaceItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ActiveCellGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OwnActiveCellGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LevelCellGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  explicit LevelCellGroupComputeFunctor(Integer level)
  : m_level(level)
  {}

 public:

  void executeFunctor() override;

 private:

  Integer m_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OwnLevelCellGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  explicit OwnLevelCellGroupComputeFunctor(Integer level)
  : m_level(level)
  {}

 public:

  void executeFunctor() override;

 private:

  Integer m_level;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ActiveFaceItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OwnActiveFaceItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class InnerActiveFaceItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OuterActiveFaceItemGroupComputeFunctor
: public ItemGroupComputeFunctor
{
 public:

  void executeFunctor() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
