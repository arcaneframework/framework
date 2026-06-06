// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPhysicalUnitConverter.h                                    (C) 2000-2025 */
/*                                                                           */
/* Interface of a unit converter.                                            */
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
 * \brief Interface of a unit converter.
 *
 * The converter is created via IPhysicalUnitSystem::createConverter().
 */
class ARCANE_CORE_EXPORT IPhysicalUnitConverter
{
 public:

  virtual ~IPhysicalUnitConverter() = default; //!< Releases resources.

 public:

  //! Returns the converted value of \a value.
  virtual Real convert(Real value) = 0;

  //! Returns the converted values of \a input_values in \a output_values.
  virtual void convert(RealConstArrayView input_values,
                       RealArrayView output_values) = 0;

  //! Starting unit
  virtual IPhysicalUnit* fromUnit() = 0;

  //! Target unit
  virtual IPhysicalUnit* toUnit() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
