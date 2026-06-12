// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FlexLMTools.h                                               (C) 2000-2026 */
/*                                                                           */
/* FlexLM Protection Management.                                             */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_UTILS_FLEXLMTOOLS_H_
#define ARCANE_UTILS_FLEXLMTOOLS_H_

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Exception.h"

#include "arcane/core/ArcaneVersion.h"

#include <map>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelSuperMng;
class TraceInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! License exception
class ARCANE_IMPL_EXPORT LicenseErrorException
: public Exception
{
 public:

  LicenseErrorException(const String& where);
  LicenseErrorException(const TraceInfo& where);
  LicenseErrorException(const String& where, const String& message);
  LicenseErrorException(const TraceInfo& where, const String& message);
  ~LicenseErrorException() ARCANE_NOEXCEPT {}

 public:

  virtual void explain(std::ostream& m) const;
  virtual void write(std::ostream& o) const;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! FlexLM manager
/*! Singleton for recording all taken feature licenses
 *
 *  The checks are performed by the master node (commRank==0)
 *  To verify the validity of this check, either the ArcaneMasterFlexLM
 *  feature must be available (which does not lock other nodes of the parallel
 *  execution on the license node) or all nodes must have ArcaneCore authorization.
 *  This is tested in the init() phase.
 */
class ARCANE_IMPL_EXPORT FlexLMMng
{
 private:

  //! Constructor
  FlexLMMng();

  //! Destructor
  virtual ~FlexLMMng() {}

 public:

  //! Access to the singleton
  static FlexLMMng* instance();

 public:

  //! Initializes the license manager
  void init(IParallelSuperMng* parallel_super_mng);

  //! Sets a new license check periodicity
  /*! The default value is 120s.
   * if t == -1     : disables periodic checking
   * if 0 <= t < 30 : the value is ignored
   * if t >= 30     : sets a new check periodicity
   */
  void setCheckInterval(const Integer t = 120);

  //! Tests the presence of a static feature
  /*! This feature will not use a license token.
   * \param do_fatal indicates whether to generate an error if unavailable
   * \return 0 if no error */
  bool checkLicense(const String name, const Real version, bool do_fatal = true) const;

  //! Requests the allocation of \param nb_licenses licenses for the feature \param name
  /*! The requested licenses are independent of the number of processors
   *  \param nb_licenses defaults to 1
   *  \return 0 if no error */
  void getLicense(const String name, const Real version, Integer nb_licenses = 1);

  //! Releases the licenses for the feature \param name
  /*! \param nb_licenses is 0 if all licenses should be released
   *  \return 0 if no error */
  void releaseLicense(const String name, Integer nb_licenses = 0);

  //! Releases all allocated licenses
  /*! \return 0 if no error */
  void releaseAllLicenses();

  //! Return info on feature
  String featureInfo(const String name, const Real version) const;

 private:

  typedef std::map<String, Integer> FeatureMapType;
  FeatureMapType m_features;
  static FlexLMMng* m_instance;
  IParallelSuperMng* m_parallel_super_mng;
  bool m_is_master; //!< Is this host the master for checks?
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Wrapper to access FlexLMMng for a given set of features
template <typename FeatureModel>
class FlexLMTools
{
 public:

  //! Constructor
  FlexLMTools() {}

  //! Destructor
  virtual ~FlexLMTools() {}

 public:

  //! Tests the availability of a feature
  /*! \return true if no error */
  bool checkLicense(typename FeatureModel::eFeature feature, const bool do_fatal) const
  {
    const String name = FeatureModel::getName(feature);
    const Real version = FeatureModel::getVersion(feature);
    return FlexLMMng::instance()->checkLicense(name, version, do_fatal);
  }

  //! Tests the availability of a feature on a maximum version
  /*! The version can be used to test a quantity; e.g., 3 for a maximum of 3 components
   * \return true if no error */
  bool checkLicense(typename FeatureModel::eFeature feature, const Real version, const bool do_fatal) const
  {
    const String name = FeatureModel::getName(feature);
    return FlexLMMng::instance()->checkLicense(name, version, do_fatal);
  }

  //! Requests the allocation of \param nb_licenses for the feature \param feature
  /*! \return 0 if no error */
  void getLicense(typename FeatureModel::eFeature feature, Integer nb_licenses = 1)
  {
    const String name = FeatureModel::getName(feature);
    const Real version = FeatureModel::getVersion(feature);
    return FlexLMMng::instance()->getLicense(name, version, nb_licenses);
  }

  //! Releases \param nb_licenses for the feature \param feature
  /*! \return 0 if no error */
  void releaseLicense(typename FeatureModel::eFeature feature, Integer nb_licenses = 0)
  {
    const String name = FeatureModel::getName(feature);
    return FlexLMMng::instance()->releaseLicense(name, nb_licenses);
  }

  //! Return info on feature
  String featureInfo(typename FeatureModel::eFeature feature) const
  {
    const String name = FeatureModel::getName(feature);
    const Real version = FeatureModel::getVersion(feature);
    return FlexLMMng::instance()->featureInfo(name, version);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
