// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeHistoryTestModule.cc                                    (C) 2000-2018 */
/*                                                                           */
/* Module de test de 'ITimeHistoryMng'.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ITimeLoopMng.h"
#include "arcane/ITimeHistoryMng.h"
#include "arcane/ITimeHistoryTransformer.h"
#include "arcane/ITimeHistoryCurveWriter2.h"

#include "arcane/tests/TimeHistoryTest_axl.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test de sous-maillage dans Arcane.
 */
class TimeHistoryTestModule
: public ArcaneTimeHistoryTestObject
{
 public:

  explicit TimeHistoryTestModule(const ModuleBuildInfo& mb);
  ~TimeHistoryTestModule() override;

 public:

  VersionInfo versionInfo() const override { return VersionInfo(1,0,0); }

 public:

  void initLoop() override;
  void exitLoop() override;
  void computeLoop() override;

 private:
  class CurveValues
  {
   public:
    CurveValues(const UniqueArray<Int32>& ax,const UniqueArray<Real>& ay)
    : x(ax), y(ay){}
    Int32UniqueArray x;
    RealUniqueArray y;
  };
  class Transformer
  : public TraceAccessor
  , public ITimeHistoryTransformer
  {
   public:
    Transformer(TimeHistoryTestModule* tm)
    : TraceAccessor(tm->traceMng()), m_module(tm){}
   public:
    void transform(CommonInfo& infos,RealSharedArray values) override
    {
      ARCANE_UNUSED(infos);
      ARCANE_UNUSED(values);
      info() << "TRANSFORM: name=" << infos.name;
      auto& c = m_module->m_curves;
      auto x = m_module->m_curves.find(infos.name);
      if (x!=c.end()){
        info() << "TRANSFORM2: name=" << infos.name;
        CurveValues& cv = x->second;
        info() << "NB_POINT x=" << cv.x.size() << " y=" << cv.y.size();
        // Multiplie les Y par deux.
        ArrayView<Real> y = cv.y;
        for( Integer k=0, kn=y.size(); k<kn; ++k )
          y[k] = y[k] * 2.0;
        // Modifie en retour les valeurs
        values = y;
        infos.iterations = cv.x;
      }
    }
    void transform(CommonInfo& infos,Int32SharedArray values) override
    { ARCANE_UNUSED(infos); ARCANE_UNUSED(values); }
    void transform(CommonInfo& infos,Int64SharedArray values) override
    { ARCANE_UNUSED(infos); ARCANE_UNUSED(values); }
   private:
    TimeHistoryTestModule* m_module;
  };

  class Visitor
  : public TraceAccessor
  , public ITimeHistoryCurveWriter2
  {
   public:
    Visitor(TimeHistoryTestModule* tm)
    : TraceAccessor(tm->traceMng()), m_thm(tm->subDomain()->timeHistoryMng()),
      m_module(tm){}
   public:
    void build() override {}
    void beginWrite(const TimeHistoryCurveWriterInfo& infos) override
    { ARCANE_UNUSED(infos); }
    void endWrite() override {}
    void writeCurve(const TimeHistoryCurveInfo& infos) override
    {
      info() << "MY_CURVE=" << infos.name();
      if (infos.name().length()==6){
        String new_name = String("New") + infos.name();
        info() << "ADD_NEW_CURVE: " << new_name;
        m_thm->addValue(new_name,1.0);
        // Sauve les valeurs de la courbe d'origine pour pouvoir les
        // modifier lors de la transformation.
        m_module->m_curves.insert(std::make_pair(new_name,CurveValues(infos.iterations(),infos.values())));
      }
    }
    String name() const override { return "Visitor"; }
    void setOutputPath(const String& path) override { m_output_path = path; }
    String outputPath() const override { return m_output_path; }
   private:
    String m_output_path;
    ITimeHistoryMng* m_thm;
    TimeHistoryTestModule* m_module;
  };

  std::map<String,CurveValues> m_curves;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_TIMEHISTORYTEST(TimeHistoryTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


TimeHistoryTestModule::
TimeHistoryTestModule(const ModuleBuildInfo& mb)
: ArcaneTimeHistoryTestObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


TimeHistoryTestModule::
~TimeHistoryTestModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryTestModule::
initLoop()
{
  info() << "INIT LOOP";
  m_global_deltat = 1.0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryTestModule::
exitLoop()
{
  info() << "EXIT LOOP";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeHistoryTestModule::
computeLoop()
{
  Integer nb_iter = subDomain()->commonVariables().globalIteration();
  bool do_stop = false;
  if (nb_iter>100){
    subDomain()->timeLoopMng()->stopComputeLoop(true);
    do_stop = true;
  }

  Real x = ((Real)nb_iter) * 1.50;
  x = math::sqrt(x);

  m_global_deltat = m_global_deltat() + 0.01;
  // Ajoute des valeurs Ã  chaque itÃ©ration
  ITimeHistoryMng* thm = subDomain()->timeHistoryMng();
  thm->addValue("Curve1",x);
  thm->addValue("Curve2",math::log(x));
  for( Integer i=3; i<45; ++i ){
    if ((nb_iter%i)==0)
      thm->addValue(String("Curve")+i,((Real)x+(Real)i)*2.3);
  }

  // En fin de calcul, rÃ©cupÃ¨re les courbes et applique une transformation
  // pour les 10 premiÃ¨res courbes. La transformation consiste Ã  crÃ©er une
  // nouvelle courbe dont les valeurs sont deux fois celle de la courbe
  // d'origine.
  if (do_stop){
    Visitor v(this);
    thm->dumpCurves(&v);

    Transformer t(this);
    thm->applyTransformation(&t);
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
