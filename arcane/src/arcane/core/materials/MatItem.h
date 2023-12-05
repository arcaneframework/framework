// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItem.h                                                   (C) 2000-2023 */
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
    _checkLevel(item_base._internal(),LEVEL_MATERIAL);
#endif
  }

  explicit ARCCORE_HOST_DEVICE MatCell(const ComponentCell& item)
  : MatCell(item.itemBase())
  {
  }

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  ARCCORE_HOST_DEVICE MatCell(ComponentItemInternal* internal)
  : MatCell(matimpl::ConstituentItemBase (internal))
  {
  }

  MatCell() = default;

 public:

  //! Maille milieu auquel cette maille matériau appartient.
  inline EnvCell envCell();

  //! Maille arcane
  Cell globalCell() const
  {
    return Cell(m_internal->globalItemBase());
  }

  //! Materiau associé
  IMeshMaterial* material() const { return _material(); }

  //! Materiau utilisateur associé
  IUserMeshMaterial* userMaterial() const { return _material()->userMaterial(); }

  //! Identifiant du matériau
  Int32 materialId() const { return m_internal->componentId(); }

 private:
  
  IMeshMaterial* _material() const { return static_cast<IMeshMaterial*>(m_internal->component()); }
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
    _checkLevel(item_base._internal(),LEVEL_ENVIRONMENT);
#endif
  }
  explicit ARCCORE_HOST_DEVICE EnvCell(const ComponentCell& item)
  : EnvCell(item.itemBase())
  {
  }
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  explicit ARCCORE_HOST_DEVICE EnvCell(ComponentItemInternal* internal)
  : EnvCell(matimpl::ConstituentItemBase(internal))
  {
  }
  EnvCell() = default;

 public:

  // Nombre de matériaux du milieu présents dans la maille
  Int32 nbMaterial() const { return m_internal->nbSubItem(); }

  //! Maille arcane
  Cell globalCell() const { return Cell(m_internal->globalItemBase()); }

  //! Maille contenant les infos sur tous les milieux
  AllEnvCell allEnvCell() const;

  //! i-ème maille matériau de cette maille
  inline MatCell cell(Integer i)
  {
    return matimpl::ConstituentItemBase(m_internal->_firstSubItem() + i);
  }

  //! Milieu associé
  IMeshEnvironment* environment() const { return _environment(); }

  //! Identifiant du milieu
  ARCCORE_HOST_DEVICE Int32 environmentId() const { return m_internal->componentId(); }

 private:
  
  IMeshEnvironment* _environment() const { return static_cast<IMeshEnvironment*>(m_internal->component()); }
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
    _checkLevel(item_base._internal(),LEVEL_ALLENVIRONMENT);
#endif
  }

  explicit ARCCORE_HOST_DEVICE AllEnvCell(const ComponentCell& item)
  : AllEnvCell(item.itemBase())
  {
  }

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  explicit ARCCORE_HOST_DEVICE AllEnvCell(ComponentItemInternal* internal)
  : AllEnvCell(matimpl::ConstituentItemBase(internal))
  {
  }

  AllEnvCell() = default;

 public:

  //! Nombre de milieux présents dans la maille
  Int32 nbEnvironment() { return m_internal->nbSubItem(); }
  
  //! Maille arcane standard
  Cell globalCell() const { return Cell(m_internal->globalItemBase()); }

  //! i-ème maille milieu
  EnvCell cell(Int32 i) const
  {
    return EnvCell(matimpl::ConstituentItemBase(m_internal->_firstSubItem() + i));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline EnvCell MatCell::
envCell()
{
  return EnvCell(m_internal->_superItemBase());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline AllEnvCell EnvCell::
allEnvCell() const
{
  return AllEnvCell(m_internal->_superItemBase());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
