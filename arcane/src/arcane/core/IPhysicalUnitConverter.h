// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPhysicalUnitConverter.h                                    (C) 2000-2025 */
/*                                                                           */
/* Interface d'un convertisseur d'unité.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPHYSICALUNITCONVERTER_H
#define ARCANE_CORE_IPHYSICALUNITCONVERTER_H
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
 * \brief Interface d'un convertisseur d'unité.
 *
 * Le convertisseur est créé via IPhysicalUnitSystem::createConverter().
 */
class ARCANE_CORE_EXPORT IPhysicalUnitConverter
{
 public:

  virtual ~IPhysicalUnitConverter() = default; //!< Libère les ressources.

 public:

  //! Retourne la valeur convertie de \a value.
  virtual Real convert(Real value) = 0;

  //! Retourne dans \a output_values les valeurs converties de \a input_values.
  virtual void convert(RealConstArrayView input_values,
                       RealArrayView output_values) = 0;

  //! Unité de départ
  virtual IPhysicalUnit* fromUnit() = 0;

  //! Unité d'arrivée
  virtual IPhysicalUnit* toUnit() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

