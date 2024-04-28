// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItem.h                                                   (C) 2000-2024 */
/*                                                                           */
/* Entités matériau et milieux.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATITEM_H
#define ARCANE_CORE_MATERIALS_MATITEM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Item.h"

#include "arcane/core/materials/ComponentItem.h"
#include "arcane/core/materials/ComponentItemInternal.h"
#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/IMeshEnvironment.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Représente un matériau d'une maille multi-matériau.
 *
 * Cette objet représente un matériau d'une maille multi-matériau.
 *
 * Il existe une maille spéciale, appelée maille nulle, pour laquelle
 * null() est vrai et qui représente une maille invalide. Dans le
 * cas de la maille invalide, il ne faut appeler aucune des autres
 * méthode de la classe sous peine de provoquer un plantage.
 *
 * \warning Ces mailles sont invalidées dès que la liste des mailles d'un
 * matériau ou d'un milieux change. Il ne faut donc pas
 * conservér une maille de ce type entre deux changements de cette liste.
 */
class MatCell
: public ComponentCell
{
 public:

  ARCCORE_HOST_DEVICE MatCell(const matimpl::ConstituentItemBase& item_base)
  : ComponentCell(item_base)
  {
#ifdef ARCANE_CHECK
    _checkLevel(item_base,LEVEL_MATERIAL);
#endif
  }

  explicit ARCCORE_HOST_DEVICE MatCell(const ComponentCell& item)
  : MatCell(item.constituentItemBase())
  {
  }

  MatCell() = default;

 public:

  //! Maille milieu auquel cette maille matériau appartient.
  ARCCORE_HOST_DEVICE inline EnvCell envCell() const;

  //! Materiau associé
  IMeshMaterial* material() const { return _material(); }

  //! Materiau utilisateur associé
  IUserMeshMaterial* userMaterial() const { return _material()->userMaterial(); }

  //! Identifiant du matériau
  ARCCORE_HOST_DEVICE Int32 materialId() const { return componentId(); }

 private:

  IMeshMaterial* _material() const { return static_cast<IMeshMaterial*>(component()); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Maille arcane d'un milieu.
 *
 * Une telle maille contient les informations sur les matériaux
 * d'un milieu pour une maille donnée.
 *
 * Il existe une maille spéciale, appelée maille nulle, pour laquelle
 * null() est vrai et qui représente une maille invalide. Dans le
 * cas de la maille invalide, il ne faut appeler aucune des autres
 * méthode de la classe sous peine de provoquer un plantage.
 *
 * \warning Ces mailles sont invalidées dès que la liste des mailles d'un
 * matériau ou d'un milieux change. Il ne faut donc pas
 * conserver une maille de ce type entre deux changements de cette liste.
 */
class EnvCell
: public ComponentCell
{
 public:

  explicit ARCCORE_HOST_DEVICE EnvCell(const matimpl::ConstituentItemBase& item_base)
  : ComponentCell(item_base)
  {
#ifdef ARCANE_CHECK
    _checkLevel(item_base,LEVEL_ENVIRONMENT);
#endif
  }
  explicit ARCCORE_HOST_DEVICE EnvCell(const ComponentCell& item)
  : EnvCell(item.constituentItemBase())
  {
  }
  EnvCell() = default;

 public:

  // Nombre de matériaux du milieu présents dans la maille
  ARCCORE_HOST_DEVICE Int32 nbMaterial() const { return nbSubItem(); }

  //! Maille contenant les infos sur tous les milieux
  ARCCORE_HOST_DEVICE inline AllEnvCell allEnvCell() const;

  //! i-ème maille matériau de cette maille
  ARCCORE_HOST_DEVICE inline MatCell cell(Integer i) const { return _subItemBase(i); }

  //! Milieu associé
  IMeshEnvironment* environment() const { return _environment(); }

  //! Identifiant du milieu
  ARCCORE_HOST_DEVICE Int32 environmentId() const { return componentId(); }

  //! Enumérateur sur les mailles matériaux de cette maille
  ARCCORE_HOST_DEVICE CellMatCellEnumerator subMatItems() const
  {
    return CellMatCellEnumerator(*this);
  }

 private:

  IMeshEnvironment* _environment() const { return static_cast<IMeshEnvironment*>(component()); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Maille arcane avec info matériaux et milieux.
 *
 * Une telle maille contient les informations sur les milieux
 * pour une maille donnée. Elle permet par exemple de connaitre le nombre
 * de milieux et pour chacun la liste des matériaux.
 *
 * \warning Ces mailles sont invalidées dès que la liste des mailles d'un
 * matériau ou d'un milieux change. Il ne faut donc pas
 * conservér une maille de ce type entre deux changements de cette liste.
 */
class AllEnvCell
: public ComponentCell
{
 public:

  explicit ARCCORE_HOST_DEVICE AllEnvCell(const matimpl::ConstituentItemBase& item_base)
  : ComponentCell(item_base)
  {
#if defined(ARCANE_CHECK)
    _checkLevel(item_base,LEVEL_ALLENVIRONMENT);
#endif
  }

  explicit ARCCORE_HOST_DEVICE AllEnvCell(const ComponentCell& item)
  : AllEnvCell(item.constituentItemBase())
  {
  }

  AllEnvCell() = default;

 public:

  //! Nombre de milieux présents dans la maille
  ARCCORE_HOST_DEVICE Int32 nbEnvironment() const { return nbSubItem(); }

  //! i-ème maille milieu
  EnvCell cell(Int32 i) const { return EnvCell(_subItemBase(i)); }

  //! Enumérateur sur les mailles milieux de cette maille
  ARCCORE_HOST_DEVICE CellEnvCellEnumerator subEnvItems() const
  {
    return CellEnvCellEnumerator(*this);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_HOST_DEVICE inline EnvCell MatCell::
envCell() const
{
  return EnvCell(_superItemBase());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_HOST_DEVICE inline AllEnvCell EnvCell::
allEnvCell() const
{
  return AllEnvCell(_superItemBase());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
