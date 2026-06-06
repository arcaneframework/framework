// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryTransformer.h                                   (C) 2000-2025 */
/*                                                                           */
/* Interface of an object transforming history curves.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMEHISTORYTRANSFORMER_H
#define ARCANE_CORE_ITIMEHISTORYTRANSFORMER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup StandardService
 * \brief Interface of an object transforming history curves.
 *
 * Classes implementing this interface can transform the
 * time history curves. This allows, for example, modifying
 * the points of the curves or removing them. 
 *
 * Usage is done via the call to ITimeHistoryMng::applyTransformation(), which
 * will call the transform() method of this instance for each curve.
 *
 * It is permitted to change the number of elements of the curve, but
 * infos.iterations.size()*infos.sub_size and values.size() must have the same number
 * of elements. It is not permitted to change the name of the curve or
 * the number of values per iteration (sub_size).
 */
class ARCANE_CORE_EXPORT ITimeHistoryTransformer
{
 public:

  //! Common info for each curve
  class CommonInfo
  {
   public:

    //! Name of the curve
    String name;
    //! List of iterations
    Int32SharedArray iterations;
    //! Number of values per curve
    Integer sub_size = 0;
  };

 public:

  virtual ~ITimeHistoryTransformer() = default; //!< Frees resources

 public:

  //! Applies the transformation for a curve with \a Real values
  virtual void transform(CommonInfo& infos, RealSharedArray values) = 0;

  //! Applies the transformation for a curve with \a Int32 values
  virtual void transform(CommonInfo& infos, Int32SharedArray values) = 0;

  //! Applies the transformation for a curve with \a Int64 values
  virtual void transform(CommonInfo& infos, Int64SharedArray values) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
