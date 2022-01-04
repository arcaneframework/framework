// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataInternal.h                                             (C) 2000-2021 */
/*                                                                           */
/* Partie interne à Arcane de IData.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IDATAINTERNAL_H
#define ARCANE_CORE_INTERNAL_IDATAINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée tableau d'un type \a T
 */
template <class DataType>
class IArrayDataInternalT
{
 public:

  virtual ~IArrayDataInternalT() = default;

  //! Réserve de la mémoire pour \a new_capacity éléments
  virtual void reserve(Integer new_capacity) =0;

  //! Conteneur associé à la donnée.
  virtual Array<DataType>& _internalDeprecatedValue() = 0;

  //! Capacité allouée par le conteneur
  virtual Integer capacity() const =0;

  //! Libère la mémoire additionnelle éventuellement allouée
  virtual void shrink() const =0;

  //! Redimensionne le conteneur.
  virtual void resize(Integer new_size) =0;

  //! Vide le conteneur et libère la mémoire alloué.
  virtual void dispose() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée tableau bi-dimensionnel d'un type \a T
 */
template <class DataType>
class IArray2DataInternalT
{
 public:

  virtual ~IArray2DataInternalT() = default;

  //! Réserve de la mémoire pour \a new_capacity éléments
  virtual void reserve(Integer new_capacity) =0;

  //! Conteneur associé à la donnée.
  virtual Array2<DataType>& _internalDeprecatedValue() = 0;

  //! Libère la mémoire additionnelle éventuellement allouée
  virtual void shrink() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
