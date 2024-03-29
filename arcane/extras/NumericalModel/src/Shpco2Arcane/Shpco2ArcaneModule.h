﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef SHPCO2ARCANE_SHPCO2ARCANE_SHPCO2ARCANEMODULE_H
#define SHPCO2ARCANE_SHPCO2ARCANE_SHPCO2ARCANEMODULE_H
/* Author : havep at Tue Apr  7 11:19:04 2009
 * Generated by createNew
 */

using namespace ArcGeoSim::Surface ;

#include "Mesh/GroupCreator/IGroupCreator.h"
#include "Appli/IAppServiceMng.h"
#include "TimeUtils/TimeMngBase.h"
#include "ExpressionParser/IExpressionParser.h"
#include "Numerics/Expressions/IExpressionMng.h"

#include "Shpco2Arcane_axl.h"

using namespace Arcane ;

// Pré-déclarations
class IGeometryMng;
class IGeometryPolicy;

//! Module de l'application Shpco2Arcane
class Shpco2ArcaneModule
  : public ArcaneShpco2ArcaneObject
  , public IAppServiceMng
  , private TimeMngBase

{
 public:
  /** Constructeur de la classe */
  Shpco2ArcaneModule(const Arcane::ModuleBuildInfo& mbi);

  /** Destructeur de la classe */
  virtual ~Shpco2ArcaneModule();

protected:
  void initializeAppServiceMng();

public:
  void prepareInit();
  void init();
  void continueInit();
  void startTimeStep();
  void endTimeStep();
  void validate();

  /** Retourne le numéro de version du module */
  virtual Arcane::VersionInfo versionInfo() const { return Arcane::VersionInfo(1,0,0); }

private :
  bool m_initialized;
  IGeometryMng * m_geometry_mng;
  IGeometryPolicy * m_geometry_policy;
};

#endif /* SHPCO2ARCANE_SHPCO2ARCANE_SHPCO2ARCANEMODULE_H */
