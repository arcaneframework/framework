// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * HyodaIceT.h                                                 (C) 2000-2012 *
 *                                                                           *
 * Header du debugger hybrid.                                                *
 *****************************************************************************/
#ifndef ARCANE_HYODA_ICE_T_H
#define ARCANE_HYODA_ICE_T_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class Hyoda;
class HyodaTcp;
class HyodaMix;

/******************************************************************************
 * Hyoda CLASS
 *****************************************************************************/
class ARCANE_HYODA_EXPORT HyodaIceT
: public TraceAccessor
{
public:
  HyodaIceT(Hyoda*, ISubDomain*, ITraceMng*, unsigned int, unsigned int, HyodaTcp*);
  ~HyodaIceT();
public:
  int renderSize(void);
  void render(void);
  void setLeftRightBottomTop(void);
  void drawGL(void);
  void sxyzip(double*);
  void setColor(double,double,double,Real3&);
  void iceColorMinMax(Real &min, Real &max);
  IVariable *getVariable(void){return m_variable;}
  void setVariable(IVariable *var){m_variable=var;}
  void checkOglError(void);
  void checkIceTError(void);
private:
  void initGL(void);
  void drawArcaneMesh(void);
  inline void drawArcPoints(const VariableItemReal3&,VariableItemReal&,double min,double max);
  inline void drawArcLines(const VariableItemReal3&,VariableItemReal&);
  inline void drawArcPolygons(void);
  inline void drawArcPolygonsWithoutVariable(void);
private:
  void LowerTriangleImage(void*);
  void UpperTriangleImage(void*);
  void gluSphere(float, int, int);
  void renderSphereConeTorus(void);
  void gluCone(float, float, int, int);
  void gluTorus(float, float, int, int);
private:
  Hyoda *hyoda;
  ISubDomain *m_sub_domain;
  struct IceTContextStruct *icetContext;
  struct IceTCommunicatorStruct *icetCommunicator;
  struct osmesa_context *osMesaContext;
  Integer rank;
  Integer num_proc;
  Integer m_screen_width;
  Integer m_screen_height;
  void *m_image_buffer;
  //! Variable dans laquelle QHyoda renseigne le point de vue de l'image à render'er
  // ainsi que l'index de la variable arcane
  // ainsi que l'index du plugin de dessin (global|material|environements)
  Real m_pov_sxyzip[6];
  double scale;
  double rot_x, rot_y, rot_z;
  int variable_index;
  int variable_plugin;
  double lrbtnf[6];
  HyodaTcp *m_tcp;
  HyodaMix *m_hyoda_mix;
  IVariable* m_variable;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  // ARCANE_HYODA_ICE_T_H
