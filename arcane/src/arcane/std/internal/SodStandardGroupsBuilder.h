// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SodStandardGroupsBuilder.h                                  (C) 2000-2020 */
/*                                                                           */
/* Création des groupes pour les cas test de tube à choc de Sod.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_SODSTANDARDGROUPSBUILDER_H
#define ARCANE_STD_INTERNAL_SODSTANDARDGROUPSBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour créer les groupes standards pour un tube à choc de sod.
 *
 * Les groupes créés sont les groupes de faces correspondants aux côtés du
 * maillages (XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX), les groupes de mailles à gauche (ZG)
 * et à droite (ZD) le long de l'axe des X et pour le groupe de droite
 *  la partie en haut (ZD_HAUT) et en bas (ZD_BAS).
 *
 * \sa SodMeshGenerator
 */
class SodStandardGroupsBuilder
: public TraceAccessor
{
 public:
  explicit SodStandardGroupsBuilder(ITraceMng* tm)
  : TraceAccessor(tm){}
 public:
  void generateGroups(IMesh* mesh,Real3 min_pos,Real3 max_pos,Real middle_x,Real middle_height);
 private:
  void _createFaceGroup(IMesh* mesh,const String& name,Int32ConstArrayView faces_lid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
