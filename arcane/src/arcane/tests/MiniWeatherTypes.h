// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MiniWeatherTypes.h                                          (C) 2000-2021 */
/*                                                                           */
/* Types pour le test 'MiniWeather'                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANETEST_MINIWEATHERTYPE_H
#define ARCANETEST_MINIWEATHERTYPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest::MiniWeather
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMiniWeatherService
{
 public:

  virtual ~IMiniWeatherService() = default;

 public:

  virtual void init(IAcceleratorMng* am,Int32 nb_x,Int32 nb_z,Real final_time) = 0;
  virtual bool loop() = 0;
  /*!
   * \brief Point d'entrée de fin d'exécution.
   * Remplit \a reduced_values avec les valeurs réduites des 4 variables
   * principales.
   */
  virtual void exit(RealArrayView reduced_values) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest::MiniWeather

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

