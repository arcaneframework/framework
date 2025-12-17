// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryMng2.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Module gérant un historique de valeurs (Version 2).                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/ITimeHistoryMng.h"
#include "arcane/IIOMng.h"
#include "arcane/CommonVariables.h"
#include "arcane/ISubDomain.h"
#include "arcane/Directory.h"
#include "arcane/AbstractModule.h"
#include "arcane/EntryPoint.h"
#include "arcane/ObserverPool.h"
#include "arcane/IVariableMng.h"
#include "arcane/CaseOptionsMain.h"
#include "arcane/IParallelMng.h"
#include "arcane/ITimeHistoryCurveWriter2.h"
#include "arcane/ITimeHistoryTransformer.h"
#include "arcane/XmlNode.h"
#include "arcane/XmlNodeList.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/ServiceFinder2.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/core/IMeshMng.h"

#include "arcane/datatype/DataTypeTraits.h"

#include "arcane/impl/internal/TimeHistoryMngInternal.h"
#include "arcane/core/GlobalTimeHistoryAdder.h"

#include <variant>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Ecrivain au format GNUPLOT.
 */
class GnuplotTimeHistoryCurveWriter2
: public TraceAccessor
, public ITimeHistoryCurveWriter2
{
 public:

  GnuplotTimeHistoryCurveWriter2(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

 public:

  void build() override {}
  void beginWrite(const TimeHistoryCurveWriterInfo& infos) override
  {
    m_times = infos.times();
    String path = infos.path();
    // m_output_path surcharge les infos en argument si non vide.
    if (m_output_path.empty())
      m_output_path = path;

    m_gnuplot_path = Directory(Directory(m_output_path), "gnuplot");
    // Créé le répertoire de sortie.
    if (m_gnuplot_path.createDirectory()) {
      warning() << "Can not create gnuplot curve directory '"
                << m_gnuplot_path.path() << "'";
    }
  }
  void writeCurve(const TimeHistoryCurveInfo& infos) override
  {
    String name(infos.name().clone());

    if (infos.subDomain() != NULL_SUB_DOMAIN_ID) {
      name = "SD" + String::fromNumber(infos.subDomain()) + "_" + name;
    }
    if (infos.hasSupport()) {
      name = infos.support() + "_" + name;
    }

    String sname(m_gnuplot_path.file(name));
    FILE* ofile = fopen(sname.localstr(), "w");
    if (!ofile) {
      warning() << "Can not open gnuplot curve file '" << sname << "'";
      return;
    }
    RealConstArrayView values = infos.values();
    Int32ConstArrayView iterations = infos.iterations();
    Integer nb_val = iterations.size();
    Integer sub_size = infos.subSize();
    for (Integer i = 0; i < nb_val; ++i) {
      fprintf(ofile, "%.16E", Convert::toDouble(m_times[iterations[i]]));
      for (Integer z = 0; z < sub_size; ++z)
        fprintf(ofile, " %.16E", Convert::toDouble(values[(i * sub_size) + z]));
      fprintf(ofile, "\n");
    }
    fclose(ofile);
  }

  void endWrite() override {}

  String name() const override { return "gnuplot"; }

  void setOutputPath(const String& path) override
  {
    m_output_path = path;
  }
  String outputPath() const override
  {
    return m_output_path;
  }

 private:

  String m_output_path;
  UniqueArray<Real> m_times;
  Directory m_gnuplot_path;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Gestionnaire d'un historique de valeurs.
 *
 * IMPORTANT: ce module fournit une interface aux autres modules
 * par l'intermédiare de ITimeHistoryMng. Par conséquent, il faut être
 * sur de ne pas toucher aux variables de ce module pendant un appel à une
 * des méthodes de ITimeHistoryMng.
 */
class TimeHistoryMng2
: public AbstractModule
, public CommonVariables
, public ITimeHistoryMng
{

 public:

  TimeHistoryMng2(const ModuleBuildInfo& cb, bool add_entry_points = true);
  ~TimeHistoryMng2() override = default;

 public:

  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

 public:

  void addValue(const String& name, Real value, bool end_time, bool is_local) override
  {
    m_internal->addValue(TimeHistoryAddValueArgInternal(name, end_time, (is_local ? parallelMng()->commRank() : NULL_SUB_DOMAIN_ID)), value);
  }
  void addValue(const String& name, Int64 value, bool end_time, bool is_local) override
  {
    m_internal->addValue(TimeHistoryAddValueArgInternal(name, end_time, (is_local ? parallelMng()->commRank() : NULL_SUB_DOMAIN_ID)), value);
  }
  void addValue(const String& name, Int32 value, bool end_time, bool is_local) override
  {
    m_internal->addValue(TimeHistoryAddValueArgInternal(name, end_time, (is_local ? parallelMng()->commRank() : NULL_SUB_DOMAIN_ID)), value);
  }
  void addValue(const String& name, RealConstArrayView values, bool end_time, bool is_local) override
  {
    m_internal->addValue(TimeHistoryAddValueArgInternal(name, end_time, (is_local ? parallelMng()->commRank() : NULL_SUB_DOMAIN_ID)), values);
  }
  void addValue(const String& name, Int32ConstArrayView values, bool end_time, bool is_local) override
  {
    m_internal->addValue(TimeHistoryAddValueArgInternal(name, end_time, (is_local ? parallelMng()->commRank() : NULL_SUB_DOMAIN_ID)), values);
  }
  void addValue(const String& name, Int64ConstArrayView values, bool end_time, bool is_local) override
  {
    m_internal->addValue(TimeHistoryAddValueArgInternal(name, end_time, (is_local ? parallelMng()->commRank() : NULL_SUB_DOMAIN_ID)), values);
  }

 public:

  void timeHistoryBegin() override;
  void timeHistoryEnd() override;
  void timeHistoryInit() override;
  void timeHistoryStartInit() override;
  void timeHistoryContinueInit() override;
  void timeHistoryRestore() override;
  void timeHistoryStartInitEnd();

 public:

  void addCurveWriter(ITimeHistoryCurveWriter2* writer) override;
  void removeCurveWriter(ITimeHistoryCurveWriter2* writer) override
  {
    ARCANE_CHECK_POINTER(writer);
    removeCurveWriter(writer->name());
  }
  void removeCurveWriter(const String& name) override;

 public:

  void dumpHistory(bool is_verbose) override;
  void dumpCurves(ITimeHistoryCurveWriter2* writer) override;

  bool active() const override { return m_internal->active(); }
  void setActive(bool is_active) override { m_internal->setActive(is_active); }

  bool isDumpActive() const override { return m_internal->isDumpActive(); }
  void setDumpActive(bool is_active) override { m_internal->setDumpActive(is_active); }

  bool isShrinkActive() const override { return m_internal->isShrinkActive(); }
  void setShrinkActive(bool is_active) override { m_internal->setShrinkActive(is_active); }

  void applyTransformation(ITimeHistoryTransformer* v) override;

  ITimeHistoryMngInternal* _internalApi() override { return m_internal.get(); }

 private:

  Ref<ITimeHistoryMngInternal> m_internal;
  // Ref<ITimeHistoryAdder> m_adder;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeHistoryMng2::
TimeHistoryMng2(const ModuleBuildInfo& mb, bool add_entry_points)
: AbstractModule(mb)
, CommonVariables(this)
, m_internal(makeRef(new TimeHistoryMngInternal(subDomain()->variableMng(),
                                                makeRef(new Properties(subDomain()->propertyMng(), "ArcaneTimeHistoryProperties")))))
// , m_adder(makeRef(new GlobalTimeHistoryAdder(this)))
{
  if (add_entry_points) {
    addEntryPoint(this, "ArcaneTimeHistoryBegin", &TimeHistoryMng2::timeHistoryBegin,
                  IEntryPoint::WComputeLoop, IEntryPoint::PAutoLoadBegin);
    addEntryPoint(this, "ArcaneTimeHistoryEnd", &TimeHistoryMng2::timeHistoryEnd,
                  IEntryPoint::WComputeLoop, IEntryPoint::PAutoLoadEnd);
    addEntryPoint(this, "ArcaneTimeHistoryInit", &TimeHistoryMng2::timeHistoryInit,
                  IEntryPoint::WInit, IEntryPoint::PAutoLoadBegin);
    addEntryPoint(this, "ArcaneTimeHistoryStartInit", &TimeHistoryMng2::timeHistoryStartInit,
                  IEntryPoint::WStartInit, IEntryPoint::PAutoLoadBegin);
    addEntryPoint(this, "ArcaneTimeHistoryContinueInit", &TimeHistoryMng2::timeHistoryContinueInit,
                  IEntryPoint::WContinueInit, IEntryPoint::PAutoLoadBegin);
    addEntryPoint(this, "ArcaneTimeHistoryStartInitEnd", &TimeHistoryMng2::timeHistoryStartInitEnd,
                  IEntryPoint::WStartInit, IEntryPoint::PAutoLoadEnd);
    addEntryPoint(this, "ArcaneTimeHistoryRestore", &TimeHistoryMng2::timeHistoryRestore,
                  IEntryPoint::WRestore, IEntryPoint::PAutoLoadBegin);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryStartInit()
{
  m_internal->addNowInGlobalTime();
}

/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryStartInitEnd()
{
  m_internal->updateGlobalTimeCurve();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryBegin()
{
  // Si on n'est pas actif, on ne grossit pas inutilement le m_global_times
  // qui sera copié dans la variable backupée 'm_th_global_time'
  if (isShrinkActive() && !active()) {
    // On ne fait rien
  }
  else {
    //warning() << "timeHistoryBegin " << m_global_time() << " " << m_global_times.size();
    m_internal->addNowInGlobalTime();
  }

  // Regarde s'il faut imprimer les sorties temporelles
  {
    bool force_print_thm = false;
    int th_step = subDomain()->caseOptionsMain()->writeHistoryPeriod();
    if (th_step != 0) {
      if ((globalIteration() % th_step) == 0)
        if (parallelMng()->isMasterIO() || (m_internal->isNonIOMasterCurvesEnabled() && m_internal->isMasterIOOfSubDomain()))
          force_print_thm = true;
    }
    if (subDomain()->applicationInfo().isDebug())
      force_print_thm = true;
    if (force_print_thm)
      m_internal->dumpHistory();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryEnd()
{
  m_internal->updateGlobalTimeCurve();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryInit()
{
  //warning() << "timeHistoryInit " << m_global_time() << " " << m_global_times.size();

  info(4) << "TimeHistory is MasterIO ? " << m_internal->isMasterIO();
  if (!m_internal->isMasterIO() && (!m_internal->isNonIOMasterCurvesEnabled() || !m_internal->isMasterIOOfSubDomain()))
    return;

  m_internal->editOutputPath(Directory(subDomain()->exportDirectory(), "courbes"));
  m_internal->addObservers(subDomain()->propertyMng());

  if (platform::getEnvironmentVariable("ARCANE_DISABLE_GNUPLOT_CURVES").null()) {
    ITimeHistoryCurveWriter2* gnuplot_curve_writer = new GnuplotTimeHistoryCurveWriter2(traceMng());
    m_internal->addCurveWriter(makeRef(gnuplot_curve_writer));
  }

  ServiceBuilder<ITimeHistoryCurveWriter2> builder(subDomain());
  auto writers = builder.createAllInstances();
  for (auto& wr_ref : writers) {
    ITimeHistoryCurveWriter2* cw = wr_ref.get();
    if (cw) {
      info() << "FOUND CURVE SERVICE (V2)!";
      m_internal->addCurveWriter(wr_ref);
    }
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
addCurveWriter(ITimeHistoryCurveWriter2* writer)
{
  m_internal->addCurveWriter(makeRef(writer));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryContinueInit()
{
  if (m_internal->isMasterIO() || (m_internal->isNonIOMasterCurvesEnabled() && m_internal->isMasterIOOfSubDomain()))
    m_internal->readVariables(subDomain()->meshMng(), subDomain()->defaultMesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
timeHistoryRestore()
{
  m_internal->resizeArrayAfterRestore();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
dumpHistory(bool is_verbose)
{
  ARCANE_UNUSED(is_verbose);
  m_internal->dumpHistory();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
dumpCurves(ITimeHistoryCurveWriter2* writer)
{
  m_internal->dumpCurves(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
applyTransformation(ITimeHistoryTransformer* v)
{
  m_internal->applyTransformation(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryMng2::
removeCurveWriter(const String& name)
{
  m_internal->removeCurveWriter(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT ITimeHistoryMng*
arcaneCreateTimeHistoryMng2(ISubDomain* mng)
{
  return new TimeHistoryMng2(ModuleBuildInfo(mng, "TimeHistoryMng"));
}
extern "C++" ARCANE_IMPL_EXPORT ITimeHistoryMng*
arcaneCreateTimeHistoryMng2(ISubDomain* mng, bool add_entry_points)
{
  return new TimeHistoryMng2(ModuleBuildInfo(mng, "TimeHistoryMng"), add_entry_points);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
