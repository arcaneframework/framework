// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ProfileRegion.h                                             (C) 2000-2025 */
/*                                                                           */
/* Region for profiling.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_PROFILEREGION_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_PROFILEREGION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Region for profiling.
 *
 * This class allows associating profiling information with all
 * calculation kernels executed between the constructor and the destructor of a
 * instance of this class.
 */
class ARCCORE_COMMON_EXPORT ProfileRegion
{
 public:

  //! Start a region with name \a name
  ProfileRegion(const RunQueue& queue, const String& name);

  /*!
   * \brief Start a region with name \a name and color \a color_rgb.
   *
   * The color is given in hexadecimal RGB format. For example 0xFF0000
   * indicates the color red and 0x7F00FF indicates the color purple.
   */
  ProfileRegion(const RunQueue& queue, const String& name, Int32 color_rgb);

  ~ProfileRegion();

 private:

  Impl::IRunnerRuntime* m_runtime = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
