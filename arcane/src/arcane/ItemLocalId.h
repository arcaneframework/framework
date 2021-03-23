// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLocalId.h                                               (C) 2000-2020 */
/*                                                                           */
/* Index local sur une entité du maillage.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMLOCALID_H
#define ARCANE_ITEMLOCALID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'un Item dans une variable.
 */
class ARCANE_CORE_EXPORT ItemLocalId
{
 public:
  ARCCORE_HOST_DEVICE ItemLocalId() : m_local_id(NULL_ITEM_LOCAL_ID){}
  ARCCORE_HOST_DEVICE explicit ItemLocalId(Int32 id) : m_local_id(id){}
  // La définition de ce constructeur est dans ItemInternal.h
  inline explicit ItemLocalId(ItemInternal* item);
  inline ItemLocalId(Item item);
  ARCCORE_HOST_DEVICE operator Int32() const { return m_local_id; }
  ARCCORE_HOST_DEVICE Int32 asInt32() const { return m_local_id; }
  ARCCORE_HOST_DEVICE Int32 asInteger() const { return m_local_id; }
 public:
  ARCCORE_HOST_DEVICE Int32 localId() const { return m_local_id; }
 private:
  Int32 m_local_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'un Node dans une variable.
 */
class ARCANE_CORE_EXPORT NodeLocalId
: public ItemLocalId
{
 public:
  ARCCORE_HOST_DEVICE explicit NodeLocalId(Int32 id) : ItemLocalId(id){}
  inline NodeLocalId(Node node);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une Edge dans une variable.
 */
class ARCANE_CORE_EXPORT EdgeLocalId
: public ItemLocalId
{
 public:
  ARCCORE_HOST_DEVICE explicit EdgeLocalId(Int32 id) : ItemLocalId(id){}
  inline EdgeLocalId(Edge edge);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une Face dans une variable.
 */
class ARCANE_CORE_EXPORT FaceLocalId
: public ItemLocalId
{
 public:
  ARCCORE_HOST_DEVICE explicit FaceLocalId(Int32 id) : ItemLocalId(id){}
  inline FaceLocalId(Face item);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une Cell dans une variable.
 */
class ARCANE_CORE_EXPORT CellLocalId
: public ItemLocalId
{
 public:
  ARCCORE_HOST_DEVICE explicit CellLocalId(Int32 id) : ItemLocalId(id){}
  inline CellLocalId(Cell item);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une Particle dans une variable.
 */
class ARCANE_CORE_EXPORT ParticleLocalId
: public ItemLocalId
{
 public:
  ARCCORE_HOST_DEVICE explicit ParticleLocalId(Int32 id) : ItemLocalId(id){}
  inline ParticleLocalId(Particle item);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'un DualNode dans une variable.
 */
class ARCANE_CORE_EXPORT DualNodeLocalId
: public ItemLocalId
{
 public:
  explicit DualNodeLocalId(Int32 id) : ItemLocalId(id){}
  inline DualNodeLocalId(DualNode item);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'un Link dans une variable.
 */
class ARCANE_CORE_EXPORT LinkLocalId
: public ItemLocalId
{
 public:
  explicit LinkLocalId(Int32 id) : ItemLocalId(id){}
  inline LinkLocalId(Link item);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
