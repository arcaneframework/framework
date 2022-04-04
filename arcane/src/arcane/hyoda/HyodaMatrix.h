// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
// ****************************************************************************
// * HyodaMatrix.h                                                (C) 2000-2016
// ****************************************************************************
#ifndef ARCANE_HYODA_MATRIX_H
#define ARCANE_HYODA_MATRIX_H

#define ARCANE_HYODA_MATRIX_MAX_CPU 8

// ****************************************************************************
// ****************************************************************************

ARCANE_BEGIN_NAMESPACE

// ****************************************************************************
// ****************************************************************************

class Hyoda;
class HyodaTcp;
class HyodaMix;

// ****************************************************************************
// * Hyoda Matrix
// ****************************************************************************
class HyodaMatrix: public TraceAccessor{
public:
  HyodaMatrix(Hyoda*, ISubDomain*, ITraceMng*, unsigned int, unsigned int, HyodaTcp*);
  ~HyodaMatrix();
public:
  int renderSize(void);
  void render(void);
  void setLeftRightBottomTop();
  void drawGL(void);
  void sxyzip(double*);
  void setColor(double,double,double,Real3&);
  void iceRowMinMax(int,Real &min, Real &max);
  void iceColMinMax(int,Real &min, Real &max);
  void iceValMinMax(int,Real &min, Real &max);
  void checkOglError(void);
  void checkIceTError(void);
  void setIJVal(int,int,int*,int*,double*);
private:
  void initGL(void);
  void drawMatrix(void);
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
  double lrbtnf[6];
  HyodaTcp *m_tcp;
 private:
  int m_hyoda_matrix_cpu;
  int* m_hyoda_matrix_n;
  int** m_hyoda_matrix_i;
  int** m_hyoda_matrix_j;
  double** m_hyoda_matrix_val;
};

// ****************************************************************************
// ****************************************************************************

ARCANE_END_NAMESPACE

// ****************************************************************************
// ****************************************************************************

#endif  // ARCANE_HYODA_MATRIX_H
