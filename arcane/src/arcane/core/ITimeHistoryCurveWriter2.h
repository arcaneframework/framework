// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryCurveWriter2.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface d'un écrivain d'une courbe d'un historique (Version 2).         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMEHISTORYCURVEWRITER2_H
#define ARCANE_CORE_ITIMEHISTORYCURVEWRITER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/String.h"

/*
 * \brief Indique si on autorise l'accès aux membres privés.
 * Par défaut (mars 2016) on laisse l'accès pour des raisons de compatibilité,
 * mais il faudra le supprimer.
 */
#define ARCANE_ALLOW_CURVE_WRITER_PRIVATE_ACCESS 1
#ifdef SWIG
#undef ARCANE_ALLOW_CURVE_WRITER_PRIVATE_ACCESS
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour l'écriture d'une courbe.
 */
class TimeHistoryCurveInfo
{
 public:

  TimeHistoryCurveInfo(const String& aname, Int32ConstArrayView aiterations,
                       RealConstArrayView avalues, Integer sub_size)
  : m_name(aname)
  , m_support()
  , m_has_support(false)
  , m_iterations(aiterations)
  , m_values(avalues)
  , m_sub_size(sub_size)
  , m_sub_domain(-1)
  {}

  TimeHistoryCurveInfo(const String& aname, Int32ConstArrayView aiterations,
                       RealConstArrayView avalues, Integer sub_size, Integer sub_domain)
  : m_name(aname)
  , m_support()
  , m_has_support(false)
  , m_iterations(aiterations)
  , m_values(avalues)
  , m_sub_size(sub_size)
  , m_sub_domain(sub_domain)
  {}

  TimeHistoryCurveInfo(const String& aname, const String& asupport, Int32ConstArrayView aiterations,
                       RealConstArrayView avalues, Integer sub_size)
  : m_name(aname)
  , m_support(asupport)
  , m_has_support(true)
  , m_iterations(aiterations)
  , m_values(avalues)
  , m_sub_size(sub_size)
  , m_sub_domain(-1)
  {}

  TimeHistoryCurveInfo(const String& aname, const String& asupport, Int32ConstArrayView aiterations,
                       RealConstArrayView avalues, Integer sub_size, Integer sub_domain)
  : m_name(aname)
  , m_support(asupport)
  , m_has_support(true)
  , m_iterations(aiterations)
  , m_values(avalues)
  , m_sub_size(sub_size)
  , m_sub_domain(sub_domain)
  {}

 public:

  //! Nom de la courbe
  const String& name() const { return m_name; }
  const String& support() const { return m_support; }
  bool hasSupport() const { return m_has_support; }
  //! Liste des itérations
  Int32ConstArrayView iterations() const { return m_iterations; }
  //! Liste des valeurs de la courbe
  RealConstArrayView values() const { return m_values; }
  //! Nombre de valeur par temps
  Integer subSize() const { return m_sub_size; }
  // TODO nom pas terrible
  Integer subDomain() const { return m_sub_domain; }

#if ARCANE_ALLOW_CURVE_WRITER_PRIVATE_ACCESS
 public:

#else
 private:
#endif
  String m_name;
  String m_support;
  bool m_has_support = false;
  Int32ConstArrayView m_iterations;
  RealConstArrayView m_values;
  Integer m_sub_size = 0;
  Integer m_sub_domain = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur l'écriture des courbes.
 */
class TimeHistoryCurveWriterInfo
{
 public:

  TimeHistoryCurveWriterInfo(const String& apath, RealConstArrayView atimes)
  : m_path(apath)
  , m_times(atimes)
  {}

 public:

  /*!
   * \brief Chemin ou écrire les données (sauf si surchargé spécifiquement
   * par le service via ITimeHistoryCurveWriter2::setOutputPath())
   */
  String path() const { return m_path; }
  //! Liste des temps
  RealConstArrayView times() const { return m_times; }

#if ARCANE_ALLOW_CURVE_WRITER_PRIVATE_ACCESS
 public:

#else
 private:
#endif
  String m_path;
  RealConstArrayView m_times;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface d'un écrivain d'une courbe.
 *
 * Lors de l'écriture des courbes, l'instance sera appelée comme suit:
 * \code
 * ITimeHistoryCurveWriter2* instance = ...;
 * instance->beginWrite();
 * for( const TimeHistoryCurveInfo& curveinfo : all_curves )
 *   instance->writeCurve(curveinfo);
 * instance->endWrite()
 * \endcode
 */
class ARCANE_CORE_EXPORT ITimeHistoryCurveWriter2
{
 public:

  //! Libère les ressources
  virtual ~ITimeHistoryCurveWriter2() = default;

 public:

  virtual void build() = 0;

  /*!
   * \brief Notifie un début d'écriture.
   */
  virtual void beginWrite(const TimeHistoryCurveWriterInfo& infos) = 0;

  /*!
   * \brief Notifie la fin de l'écriture.
   */
  virtual void endWrite() = 0;

  /*!
   * \brief Ecrit une courbe.
   *
   * Les infos de la courbe sont données par \a infos
   * Les valeurs sont dans le tableau \a values. \a times et \a iterations
   * contiennent respectivement le temps et le numéro de l'itération pour
   * chaque valeur.
   * \a path contient le répertoire où seront écrites les courbes
   */
  virtual void writeCurve(const TimeHistoryCurveInfo& infos) = 0;

  //! Nom de l'écrivain
  virtual String name() const = 0;

  /*!
   * \brief Répertoire de base où seront écrites les courbes.
   *
   * Si nul, c'est le répertoire spécifié lors de beginWrite()
   * qui est utilisé.
   */
  virtual void setOutputPath(const String& path) = 0;

  //! Répertoire de base où seront écrites les courbes.
  virtual String outputPath() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
