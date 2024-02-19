// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TODO                                          (C) 2000-2024 */
/*                                                                           */
/*                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ITIMEHISTORYMNGINTERNAL_H
#define ARCANE_ITIMEHISTORYMNGINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeHistoryTransformer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeHistoryMngInternal
{
 public:
  virtual ~ITimeHistoryMngInternal() = default; //!< Libère les ressources

 public:
  virtual void addNowInGlobalTime() = 0;
  virtual void updateGlobalTimeCurve() = 0;
  virtual void resizeArrayAfterRestore() = 0;

  virtual void addCurveWriter(Ref<ITimeHistoryCurveWriter2> writer) =0;
  virtual bool isShrinkActive() const =0;
  virtual void setShrinkActive(bool is_active) =0;
  virtual bool active() const =0;
  virtual void setActive(bool is_active) =0;
  virtual void setDumpActive(bool is_active) =0;
  virtual bool isDumpActive() const =0;
  virtual void dumpHistory(bool is_verbose) =0;
  virtual void dumpCurves(ITimeHistoryCurveWriter2* writer) =0;
  virtual void applyTransformation(ITimeHistoryTransformer* v) =0;
  virtual void removeCurveWriter(const String& name) =0;
  virtual void updateMetaData() =0;
  virtual void readVariables() =0;
  virtual bool isMasterIO() = 0;
  virtual bool isNonIOMasterCurvesEnabled() = 0;

  virtual void addValue(const String& name,Real value,bool end_time=true,bool is_local=false) =0;

  /*! \brief Ajoute la valeur \a value à l'historique \a name.
   *
   * Le nombre d'éléments de \a value doit être constant au cours du temps.
   * La valeur est celle au temps de fin de l'itération si \a end_time est vrai,
   * au début sinon.
   * le booleen is_local indique si la courbe est propre au process ou pas pour pouvoir écrire des courbes meme
   * par des procs non io_master quand la variable ARCANE_ENABLE_NON_IO_MASTER_CURVES
   */
  virtual void addValue(const String& name,RealConstArrayView values,bool end_time=true,bool is_local=false) =0;

  /*! \brief Ajoute la valeur \a value à l'historique \a name.
   *
   * Le nombre d'éléments de \a value doit être constant au cours du temps.
   * La valeur est celle au temps de fin de l'itération si \a end_time est vrai,
   * au début sinon.
   * le booleen is_local indique si la courbe est propre au process ou pas pour pouvoir écrire des courbes meme
   * par des procs non io_master quand la variable ARCANE_ENABLE_NON_IO_MASTER_CURVES
   */
  virtual void addValue(const String& name,Int32ConstArrayView values,bool end_time=true,bool is_local=false) =0;

  /*! Ajoute la valeur \a value à l'historique \a name.
   *
   * Le nombre d'éléments de \a value doit être constant au cours du temps.
   * La valeur est celle au temps de fin de l'itération si \a end_time est vrai,
   * au début sinon.
   * le booleen is_local indique si la courbe est propre au process ou pas pour pouvoir écrire des courbes meme
   * par des procs non io_master quand la variable ARCANE_ENABLE_NON_IO_MASTER_CURVES
   */
  virtual void addValue(const String& name,Int64ConstArrayView values,bool end_time=true,bool is_local=false) =0;

  virtual void addValue(const String& name, const String& metadata, Real value,bool end_time=true,bool is_local=false) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

