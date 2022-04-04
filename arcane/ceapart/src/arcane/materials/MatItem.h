﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItem.h                                                   (C) 2000-2013 */
/*                                                                           */
/* Entités matériau et milieux.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MATITEM_H
#define ARCANE_MATERIALS_MATITEM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Item.h"
#include "arcane/materials/ComponentItem.h"
#include "arcane/materials/MatItemInternal.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/IMeshEnvironment.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MatCell;
class EnvCell;
class AllEnvCell;

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
class ARCANE_MATERIALS_EXPORT MatCell
: public ComponentCell
{
 public:

  MatCell(ComponentItemInternal* internal)
  : ComponentCell(internal)
  {
#ifdef ARCANE_CHECK
    _checkLevel(internal,LEVEL_MATERIAL);
#endif
  }
  MatCell() {}

 public:

  //! Maille milieu auquel cette maille matériau appartient.
  inline EnvCell envCell();

  //! Maille arcane
  Cell globalCell() const
  {
    return Cell(m_internal->globalItem());
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
class ARCANE_MATERIALS_EXPORT EnvCell
: public ComponentCell
{
 public:

  explicit EnvCell(ComponentItemInternal* internal)
  : ComponentCell(internal)
  {
#ifdef ARCANE_CHECK
    _checkLevel(internal,LEVEL_ENVIRONMENT);
#endif
  }
  EnvCell() {}

 public:

  // Nombre de matériaux du milieu présents dans la maille
  Int32 nbMaterial() const { return m_internal->nbSubItem(); }

  //! Maille arcane
  Cell globalCell() const { return Cell(m_internal->globalItem()); }

  //! Maille contenant les infos sur tous les milieux
  AllEnvCell allEnvCell() const;

  //! i-ème maille matériau de cette maille
  inline MatCell cell(Integer i)
  {
    return (m_internal->firstSubItem() + i);
  }

  //! Milieu associé
  IMeshEnvironment* environment() const { return _environment(); }

  //! Identifiant du milieu
  Int32 environmentId() const { return m_internal->componentId(); }

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
class ARCANE_MATERIALS_EXPORT AllEnvCell
: public ComponentCell
{
 public:

  explicit AllEnvCell(ComponentItemInternal* internal)
  : ComponentCell(internal)
  {
#ifdef ARCANE_CHECK
    _checkLevel(internal,LEVEL_ALLENVIRONMENT);
#endif
  }

 public:

  //! Nombre de milieux présents dans la maille
  Int32 nbEnvironment() { return m_internal->nbSubItem(); }
  
  //! Maille arcane standard
  Cell globalCell() const { return Cell(m_internal->globalItem()); }

  //! i-ème maille milieu
  EnvCell cell(Integer i) const
  {
    return EnvCell(m_internal->firstSubItem() + i);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline EnvCell MatCell::
envCell()
{
  return EnvCell(m_internal->superItem());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline AllEnvCell EnvCell::
allEnvCell() const
{
  return AllEnvCell(m_internal->superItem());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
