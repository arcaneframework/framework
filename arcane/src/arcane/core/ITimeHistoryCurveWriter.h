// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryCurveWriter.h                                   (C) 2000-2025 */
/*                                                                           */
/* Interface of a historical curve writer.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMEHISTORYCURVEWRITER_H
#define ARCANE_CORE_ITIMEHISTORYCURVEWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup StandardService
 * \brief Interface of a curve writer.
 *
 * \deprecated Use the ITimeHistoryCurveWriter2 interface instead.
 */
class ITimeHistoryCurveWriter
{
 public:

  virtual ~ITimeHistoryCurveWriter() = default; //!< Frees resources

 public:

  virtual void build() = 0;

  /*!
   * \brief Writes the curve named \a name.
   *
   * The values are in the \a values array. \a times and \a iterations
   * contain respectively the time and the iteration number for
   * each value.
   * \a path contains the directory where the curves will be written
   */
  virtual void writeCurve(const IDirectory& path,
                          const String& name,
                          Int32ConstArrayView iterations,
                          RealConstArrayView times,
                          RealConstArrayView values,
                          Integer sub_size) = 0;

  //! Writer name
  virtual String name() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
