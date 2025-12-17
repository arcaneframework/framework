// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableFactory.h                                           C) 2000-2025 */
/*                                                                           */
/* Interface d'une fabrique d'une variable d'un type donné.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEFACTORY_H
#define ARCANE_CORE_IVARIABLEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une fabrique de variables.
 *
 L'instance permet une créér une variable en fonction de son type de
 données (dataType()), son genre d'entité (itemKind()), sa dimension
 (dimension()) et son tag s'il s'agit d'un tableau multiple (multiTag()).

 L'opération fullTypeName() contient le nom complet du type, obtenu
 de la manière suivante: dataType().itemKind().dimension().multiTag. Par exemple,
 pour une variable scalaire réelle aux mailles, le type complet est
 le suivant: \a "Real.Cell.0.0".
 */
class IVariableFactory
{
 public:
 
  //! Type de la fonction créant la variable
  using VariableFactoryFunc = VariableFactoryVariableRefCreateFunc;

 public:
 
  virtual ~IVariableFactory() = default;

 public:

  //! Créé une variable avec la infos \a build_info et retourne sa référence.
  virtual VariableRef* createVariable(const VariableBuildInfo& build_info) =0;

 public:

  //! Genre des variables de données de la variable créée par cette fabrique
  virtual eItemKind itemKind() const =0;

  //! Type de données de la variable créée par cette fabrique
  virtual eDataType dataType() const =0;

  //! Dimension de la variable créée par cette fabrique
  virtual Integer dimension() const =0;

  //! Tag multi.
  virtual Integer multiTag() const =0;

  //! Nom complet du type de la variable
  virtual const String& fullTypeName() const =0;

 public:

  //! Informations sur le type de la variable
  virtual VariableTypeInfo variableTypeInfo() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

