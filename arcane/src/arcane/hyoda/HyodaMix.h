// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * HyodaMix.h                                                  (C) 2000-2013 *
 *****************************************************************************/
#ifndef ARCANE_HYODA_MIX_H
#define ARCANE_HYODA_MIX_H

#include "arcane/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class Hyoda;
class IHyodaPlugin;

/******************************************************************************
 * HyodaMix CLASS
 *****************************************************************************/
class ARCANE_HYODA_EXPORT HyodaMix
: public TraceAccessor
{
public:
  HyodaMix(Hyoda*, ISubDomain*, ITraceMng*);
  ~HyodaMix(){}
public:
  void xLine2Cell(int,IVariable*,Real,Real);
public:
  void setCellOrigin(Cell);
  Int32 xCellPoints(Cell, Real3, Real,Int32);
  void xCellDrawNormal(Cell, Real3 p[4], Int32 iDst);
  void xCellDrawInterface(Cell, Int32);
public:
  int xCellBorders(Cell, Real, Real, Real);
  int xCellFill(Cell, Int32Array&, Real, Real, Real, Int32,Int32);
  int xCellFill_i2_o0(Cell, Real3 p[4], Real3 x[12], Int32Array&,Real3);
  int xCellFill_i2_o1(Cell, Real3 p[4], Real3 x[12], Int32Array&,Real3);
  int xCellFill_i3_o0(Cell, Real3 p[4], Real3 x[12], Int32Array&,Real3);
  int xCellFill_i3_o1(Cell, Real3 p[4], Real3 x[12], Int32Array&,Real3);
  int xCellFill_i3_o2(Cell, Real3 p[4], Real3 x[12], Int32Array&,Real3);
private:
  int xNrmDstSgmt2Point(Real3 p0, Real3 d0,  Real3 p1, Real3 p2, Real3 &);
private:
  Hyoda *m_hyoda;
  IHyodaPlugin *m_hPlgMats;
  IHyodaPlugin *m_hPlgEnvs;
  ISubDomain *m_sub_domain;
  IMesh* m_default_mesh;    
  VariableCellReal3 m_interface_normal;
  VariableCellArrayReal m_interface_distance;
  Real3UniqueArray m_p;
  Real3UniqueArray m_x;
  VariableNodeReal3 coords;
  VariableCellInteger m_i_origine;
  VariableCellInteger m_x_codes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  // ARCANE_HYODA_MIX_H
